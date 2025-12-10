import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, Loader2, CheckCircle2 } from 'lucide-react';
import { db } from '../db';

const Scanner = ({ onScanComplete }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const requestRef = useRef(null);

  // --- AUTO-SAVE LOGIC REFS ---
  // Tracks how many consistent frames we've seen for a specific ID
  // Structure: { [track_id]: { count: number, signature: string } }
  const scanTrackerRef = useRef(new Map());

  // Tracks IDs that have already been saved to DB to prevent double-saving
  const savedSessionsRef = useRef(new Set());

  // Persistence Refs (For smoothing UI)
  const lastDetectionsRef = useRef([]);
  const lastDetectionTimeRef = useRef(Date.now());

  // UI States
  const [isPaused, setIsPaused] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [lastSavedItem, setLastSavedItem] = useState(null);

  useEffect(() => {
    startCamera();
    connectWebSocket();

    return () => {
      stopCamera();
      if (socketRef.current) socketRef.current.close();
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, []);

  const connectWebSocket = () => {
    socketRef.current = new WebSocket('ws://localhost:8000/ws');

    socketRef.current.onopen = () => {
      console.log('✅ WebSocket Connected');
      setIsConnected(true);
      processFrame();
    };

    socketRef.current.onmessage = (event) => {
      if (isPaused) return;

      const data = JSON.parse(event.data);

      if (data.detections && data.detections.length > 0) {
        // 1. Update UI Refs
        lastDetectionsRef.current = data.detections;
        lastDetectionTimeRef.current = Date.now();

        // 2. Draw Boxes
        drawOverlay(data.detections);

        // 3. Process ALL detections for auto-save (Multi-item support)
        data.detections.forEach((det) => {
          handleAutoSaveLogic(det);
        });
      } else {
        const timeSinceLast = Date.now() - lastDetectionTimeRef.current;
        if (timeSinceLast < 200 && lastDetectionsRef.current.length > 0) {
          drawOverlay(lastDetectionsRef.current);
        } else {
          clearOverlay();
        }
      }

      requestRef.current = requestAnimationFrame(processFrame);
    };

    socketRef.current.onerror = (error) => {
      console.error('❌ WebSocket Error:', error);
    };

    socketRef.current.onclose = () => {
      console.log('❌ WebSocket Disconnected');
      setIsConnected(false);
    };
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (canvasRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
          }
        };
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      alert('Cannot access camera. Please grant camera permissions.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
    }
  };

  const processFrame = () => {
    // If paused, we keep the loop alive but do nothing
    if (isPaused || !videoRef.current || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      requestRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const offCanvas = document.createElement('canvas');
    const w = videoRef.current.videoWidth;
    const h = videoRef.current.videoHeight;

    if (w === 0 || h === 0) return;

    offCanvas.width = w;
    offCanvas.height = h;
    const ctx = offCanvas.getContext('2d');

    ctx.drawImage(videoRef.current, 0, 0, w, h);

    offCanvas.toBlob(
      (blob) => {
        if (blob && socketRef.current.readyState === WebSocket.OPEN) {
          socketRef.current.send(blob);
        }
      },
      'image/webp',
      0.9
    );
  };

  const clearOverlay = () => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  const drawOverlay = (detections) => {
    if (!canvasRef.current || !videoRef.current) return;
    const ctx = canvasRef.current.getContext('2d');

    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    const scaleX = canvasRef.current.width / videoRef.current.videoWidth;
    const scaleY = canvasRef.current.height / videoRef.current.videoHeight;

    detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;

      const rX = x1 * scaleX;
      const rY = y1 * scaleY;
      const rW = (x2 - x1) * scaleX;
      const rH = (y2 - y1) * scaleY;

      const isSaved = savedSessionsRef.current.has(det.track_id);

      // Draw Box
      ctx.strokeStyle = isSaved ? '#3b82f6' : det.color || '#10B981'; // Blue if saved, else standard color
      ctx.lineWidth = 4;
      ctx.strokeRect(rX, rY, rW, rH);

      // Draw Label (REMOVED CONFIDENCE)
      const text = isSaved ? `SAVED` : `${det.class} | ${det.condition}`;
      ctx.font = 'bold 24px monospace';
      const textMetrics = ctx.measureText(text);

      ctx.fillStyle = isSaved ? '#3b82f6' : det.color || '#10B981';
      ctx.fillRect(rX, rY - 35, textMetrics.width + 10, 35);

      ctx.fillStyle = det.condition === 'Clean' ? '#000000' : '#FFFFFF';
      ctx.fillText(text, rX + 5, rY - 8);
    });
  };

  // --- CORE LOGIC: MULTI-ITEM STABILITY CHECK ---
  const handleAutoSaveLogic = async (result) => {
    const trackId = result.track_id;

    // 1. Ignore if already saved
    if (savedSessionsRef.current.has(trackId)) return;

    // 2. Create a signature to detect change (e.g. "PET-Clean")
    // If the model flickers from "Clean" to "Dirty", we consider that unstable and restart.
    const currentSignature = `${result.class}-${result.condition}`;

    // 3. Get existing tracker for this ID or initialize
    let tracker = scanTrackerRef.current.get(trackId) || { count: 0, signature: currentSignature };

    // 4. Compare Signature
    if (tracker.signature === currentSignature) {
      // Prediction is STABLE, increment count
      tracker.count += 1;
    } else {
      // Prediction FLIPPED (e.g., PET -> HDPE). RESTART count.
      tracker.count = 1;
      tracker.signature = currentSignature;
    }

    // 5. Update Map
    scanTrackerRef.current.set(trackId, tracker);

    // 6. Trigger Save if Threshold Reached (15 Frames)
    // UPDATED: Changed from 10 to 15 as requested
    if (tracker.count >= 15) {
      // Mark as saved IMMEDIATELY to prevent double-fire while async DB call happens
      savedSessionsRef.current.add(trackId);

      // Save
      await saveToDatabase(result);

      // Visual Feedback
      setLastSavedItem(`${result.class} - ${result.condition}`);
      setTimeout(() => setLastSavedItem(null), 3000);

      // Haptic Feedback
      if (navigator.vibrate) navigator.vibrate(200);
    }
  };

  const saveToDatabase = async (result) => {
    const typeMap = {
      PET: 1,
      HDPE: 2,
      PVC: 3,
      LDPE: 4,
      PP: 5,
      PS: 6,
      OTHER: 7,
    };

    const plasticTypeID = typeMap[result.class] || 7;

    const newScanData = {
      plasticTypeID: plasticTypeID,
      condition: result.condition || 'Clean',
      confidence: parseFloat(result.confidence || 0),
      status: 'New',
      timestamp: new Date().toISOString(),
    };

    try {
      await db.collection.add(newScanData);
      if (onScanComplete) onScanComplete();
    } catch (e) {
      console.error('Error saving scan', e);
    }
  };

  const togglePause = () => {
    const nextState = !isPaused;
    setIsPaused(nextState);

    if (nextState) {
      // We are pausing: Freeze the actual video element
      if (videoRef.current) videoRef.current.pause();
      // NOTE: We do NOT call clearOverlay() here, so the user can see the last detection
    } else {
      // We are resuming: Play the video
      if (videoRef.current) videoRef.current.play();
    }
  };

  return (
    <div className="h-full flex flex-col relative bg-black">
      <div className="flex-1 relative overflow-hidden flex items-center justify-center">
        {!isConnected && (
          <div className="absolute top-4 left-4 z-50 bg-rose-500/90 text-white px-4 py-2 rounded-lg text-sm font-bold flex items-center gap-2 shadow-lg backdrop-blur">
            <Loader2 className="animate-spin" size={16} />
            <span>Connecting...</span>
          </div>
        )}

        {isPaused && (
          <div className="absolute inset-0 z-40 bg-black/60 backdrop-blur-sm flex items-center justify-center">
            <div className="text-white flex flex-col items-center">
              <Pause size={48} />
              <span className="font-bold text-xl mt-2">PAUSED</span>
            </div>
          </div>
        )}

        {/* Success Toast */}
        {lastSavedItem && (
          <div className="absolute top-8 left-1/2 transform -translate-x-1/2 z-50 bg-emerald-500 text-white px-6 py-3 rounded-full shadow-2xl flex items-center gap-2 animate-bounce">
            <CheckCircle2 size={24} className="text-white" />
            <span className="font-bold">Saved: {lastSavedItem}</span>
          </div>
        )}

        <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover" />
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover pointer-events-none" />
      </div>

      {/* NEW CONTROL BAR */}
      <div className="bg-white p-6 border-t border-slate-200 z-10 flex justify-center items-center pb-10">
        <button
          onClick={togglePause}
          className={`
            w-20 h-20 rounded-full flex items-center justify-center shadow-xl transition-all duration-200 transform
            ${
              !isPaused
                ? 'bg-rose-100 text-rose-600 hover:bg-rose-200 hover:scale-105 active:scale-95'
                : 'bg-emerald-100 text-emerald-600 hover:bg-emerald-200 hover:scale-105 active:scale-95'
            }
          `}
        >
          {isPaused ? <Play size={36} fill="currentColor" /> : <Pause size={36} fill="currentColor" />}
        </button>
      </div>
    </div>
  );
};

export default Scanner;
