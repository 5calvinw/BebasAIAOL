import cv2
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
import numpy as np
from collections import deque, defaultdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio

# --- CONFIGURATION ---
YOLO_PATH = "model/result/weights/best.pt"
RESNET_PATH = "model/result/weights/robust_resnet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "OTHER"]
CONDITIONS = ["Clean", "Dirty"]

# CRITICAL: Match training parameters
EXPANSION_RATIO = 0.15


# --- UPDATED MODEL DEFINITION (Matches Training) ---
class RobustResNet(nn.Module):
    def __init__(self):
        super(RobustResNet, self).__init__()
        # 1. Load Backbone
        self.backbone = models.resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 2. MATCH THE NEW TRAINING SIZE (256, not 512)
        self.hidden_size = 256

        self.head_type = nn.Sequential(
            nn.Linear(num_features, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, 7),
        )

        self.head_cond = nn.Sequential(
            nn.Linear(num_features, self.hidden_size), nn.ReLU(), nn.Dropout(0.5), nn.Linear(self.hidden_size, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head_type(features), self.head_cond(features)


# --- HELPERS ---
def load_resnet():
    print(f"Loading ResNet from {RESNET_PATH}...")
    # Update class name to match training
    model = RobustResNet().to(DEVICE)
    try:
        checkpoint = torch.load(RESNET_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        if DEVICE.type == "cuda":
            model.half()
        print("✅ ResNet Loaded Successfully")
    except Exception as e:
        print(f"❌ Error loading ResNet: {e}")
        print("   Make sure you ran the NEW training script (7_resnet_training.py)!")
    return model


def expand_box(x, y, w, h, img_w, img_h, ratio=0.15):
    center_x = x + w / 2
    center_y = y + h / 2
    new_w = w * (1 + ratio)
    new_h = h * (1 + ratio)
    new_x = center_x - new_w / 2
    new_y = center_y - new_h / 2
    x1 = max(0, int(new_x))
    y1 = max(0, int(new_y))
    x2 = min(img_w, int(new_x + new_w))
    y2 = min(img_h, int(new_y + new_h))
    return x1, y1, x2, y2


def pad_to_square(img):
    """
    CRITICAL: Matches training preprocessing.
    Pads the image with the background color to make it square.
    """
    h, w = img.shape[:2]
    if h == w:
        return img

    max_side = max(h, w)

    # Sample top-left 10x10 to get background color (Green Screen Logic)
    sample_h = min(10, h)
    sample_w = min(10, w)
    sample = img[0:sample_h, 0:sample_w]
    bg_color = np.mean(sample, axis=(0, 1)).astype(int).tolist()

    square = np.full((max_side, max_side, 3), bg_color, dtype=np.uint8)
    x_off = (max_side - w) // 2
    y_off = (max_side - h) // 2
    square[y_off : y_off + h, x_off : x_off + w] = img

    return square


class PredictionSmoother:
    def __init__(self, maxlen=5):
        self.maxlen = maxlen
        self.history_type = defaultdict(lambda: deque(maxlen=maxlen))
        self.history_cond = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, track_id, probs_type, probs_cond):
        self.history_type[track_id].append(probs_type)
        self.history_cond[track_id].append(probs_cond)

    def get_smoothed_prediction(self, track_id):
        if len(self.history_type[track_id]) == 0:
            return 0, 0, 0.0

        avg_type = torch.stack(list(self.history_type[track_id])).mean(dim=0)
        avg_cond = torch.stack(list(self.history_cond[track_id])).mean(dim=0)

        idx_type = torch.argmax(avg_type).item()
        idx_cond = torch.argmax(avg_cond).item()

        # Return confidence of the winner class
        return idx_type, idx_cond, avg_type[idx_type].item()


# --- INITIALIZATION ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Initializing Global Models...")
yolo_model = YOLO(YOLO_PATH)
resnet_model = load_resnet()

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
if DEVICE.type == "cuda":
    MEAN = MEAN.half()
    STD = STD.half()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"✅ Client Connected from {websocket.client}")

    smoother = PredictionSmoother(maxlen=5)

    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            h_frame, w_frame = frame.shape[:2]

            # Run YOLO with persistence (Tracking)
            results = yolo_model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

            detections_to_send = []
            batch_crops = []
            batch_meta = []

            for r in results:
                if r.boxes.id is None:
                    continue

                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                track_ids = r.boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box

                    # Clamp coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_frame, x2), min(h_frame, y2)

                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue

                    # 1. Expand Box (Context)
                    box_w = x2 - x1
                    box_h = y2 - y1
                    ex1, ey1, ex2, ey2 = expand_box(x1, y1, box_w, box_h, w_frame, h_frame, EXPANSION_RATIO)

                    crop = frame[ey1:ey2, ex1:ex2]

                    # 2. Pad to Square (Distortion Fix)
                    crop_padded = pad_to_square(crop)
                    crop_resized = cv2.resize(crop_padded, (224, 224))
                    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

                    batch_crops.append(crop_rgb)
                    # Store ORIGINAL box coords for display
                    batch_meta.append((x1, y1, x2, y2, track_id))

            best_result = None
            highest_conf = 0

            # Batch Inference
            if batch_crops:
                batch_np = np.stack(batch_crops)
                tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(DEVICE)
                tensor = tensor.float() / 255.0
                if DEVICE.type == "cuda":
                    tensor = tensor.half()
                tensor = (tensor - MEAN) / STD

                with torch.no_grad():
                    out_type, out_cond = resnet_model(tensor)
                    probs_type = torch.nn.functional.softmax(out_type, dim=1)
                    probs_cond = torch.nn.functional.softmax(out_cond, dim=1)

                for i, (x1, y1, x2, y2, track_id) in enumerate(batch_meta):
                    smoother.update(track_id, probs_type[i], probs_cond[i])
                    idx_type, idx_cond, conf = smoother.get_smoothed_prediction(track_id)

                    pred_type = CLASSES[idx_type]
                    pred_cond = CONDITIONS[idx_cond]

                    result_obj = {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class": pred_type,
                        "condition": pred_cond,
                        "confidence": float(conf),
                        "track_id": int(track_id),
                        "color": "#10B981" if pred_cond == "Clean" else "#EF4444",
                    }
                    detections_to_send.append(result_obj)

                    if conf > highest_conf:
                        highest_conf = conf
                        best_result = result_obj

            await websocket.send_json({"detections": detections_to_send, "best_result": best_result})

    except WebSocketDisconnect:
        print("❌ Client Disconnected")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
