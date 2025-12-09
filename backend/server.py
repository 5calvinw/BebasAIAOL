import cv2
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
import numpy as np
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from collections import deque, defaultdict
import asyncio

# --- CONFIGURATION ---
YOLO_PATH = "model/result/weights/best.pt"
RESNET_PATH = "model/result/weights/dual_head_resnet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "OTHER"]
CONDITIONS = ["Clean", "Dirty"]

# --- REMOVED TEMPERATURE SCALING ---
# 8_engine.py does not use temperature scaling. 
# We remove it here to ensure the confidence math is identical.

class PredictionSmoother:
    def __init__(self, maxlen=5):
        self.maxlen = maxlen
        self.history_type = defaultdict(lambda: deque(maxlen=maxlen))
        self.history_cond = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, track_id, probs_type, probs_cond):
        self.history_type[track_id].append(probs_type)
        self.history_cond[track_id].append(probs_cond)

    def get_smoothed_prediction(self, track_id):
        # Average probability vectors
        avg_type = torch.stack(list(self.history_type[track_id])).mean(dim=0)
        avg_cond = torch.stack(list(self.history_cond[track_id])).mean(dim=0)

        idx_type = torch.argmax(avg_type).item()
        idx_cond = torch.argmax(avg_cond).item()

        return idx_type, idx_cond, avg_type[idx_type].item()

class DualHeadResNet(nn.Module):
    def __init__(self):
        super(DualHeadResNet, self).__init__()
        self.backbone = models.resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head_type = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7),
        )

        self.head_cond = nn.Sequential(
            nn.Linear(num_features, 512), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(512, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head_type(features), self.head_cond(features)

# --- GLOBAL VARIABLES ---
yolo_model = None
resnet_model = None
mean_tensor = None
std_tensor = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_models():
    global yolo_model, resnet_model, mean_tensor, std_tensor
    
    print("⏳ Loading YOLO...")
    yolo_model = YOLO(YOLO_PATH)
    
    print("⏳ Loading ResNet...")
    resnet_model = DualHeadResNet().to(DEVICE)
    try:
        checkpoint = torch.load(RESNET_PATH, map_location=DEVICE)
        resnet_model.load_state_dict(checkpoint)
        resnet_model.eval()
        if DEVICE.type == "cuda":
            resnet_model.half()
        print("✅ Models Loaded Successfully!")
    except Exception as e:
        print(f"❌ Error loading ResNet: {e}")

    # Standard ImageNet normalization
    mean_tensor = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std_tensor = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    if DEVICE.type == "cuda":
        mean_tensor = mean_tensor.half()
        std_tensor = std_tensor.half()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    # Instantiate smoother per client connection
    smoother = PredictionSmoother(maxlen=2)
    
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            results = process_frame(frame, smoother)
            await websocket.send_json(results)
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

def process_frame(frame, smoother):
    yolo_results = yolo_model.track(frame, persist=True, verbose=False, conf=0.5, tracker="bytetrack.yaml")
    
    detections = []
    crops = []
    meta = []
    
    for r in yolo_results:
        if r.boxes.id is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = r.boxes.id.cpu().numpy().astype(int)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            track_id = int(track_ids[i])
            
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0: continue
            
            crop = cv2.resize(crop, (224, 224))
        
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            crops.append(crop)
            meta.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "track_id": track_id
            })

    if not crops:
         return {"detections": [], "best_result": None}

    tensor = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(DEVICE)
    tensor = tensor.float() / 255.0
    
    if DEVICE.type == "cuda":
        tensor = tensor.half()
    
    tensor = (tensor - mean_tensor) / std_tensor
    
    with torch.no_grad():
        out_type, out_cond = resnet_model(tensor)
        
        probs_type = torch.nn.functional.softmax(out_type, dim=1)
        probs_cond = torch.nn.functional.softmax(out_cond, dim=1)
        
    for i, m in enumerate(meta):
        track_id = m["track_id"]
        
        smoother.update(track_id, probs_type[i], probs_cond[i])

        type_idx, cond_idx, conf_val = smoother.get_smoothed_prediction(track_id)

        detections.append({
            "track_id": track_id,
            "bbox": m["bbox"],
            "class": CLASSES[type_idx],
            "condition": CONDITIONS[cond_idx],
            "confidence": float(conf_val),
            "color": "#10b981" if CONDITIONS[cond_idx] == "Clean" else "#ef4444"
        })

    best_detection = max(detections, key=lambda x: x['confidence']) if detections else None

    return {
        "detections": detections,
        "best_result": best_detection
    }