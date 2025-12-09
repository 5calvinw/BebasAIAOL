import React, { useState, useEffect, useRef } from 'react';
import { Camera, Settings, History, Loader2 } from 'lucide-react';
import { db } from '../db';
import { PLASTIC_STYLE_MAP } from '../utils/constants';

const Scanner = ({ onScanComplete }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const requestRef = useRef(null);

  const [isScanning, setIsScanning] = useState(false);
  const [liveData, setLiveData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

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
      const data = JSON.parse(event.data);
      setLiveData(data);

      if (data.detections) {
        drawOverlay(data.detections);
      }

      requestRef.current = requestAnimationFrame(processFrame);
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
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
    }
  };

  const processFrame = () => {
    if (!videoRef.current || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      requestRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const sendWidth = 1280;
    const sendHeight = 720;

    const modelCanvas = document.createElement('canvas');
    modelCanvas.width = sendWidth;
    modelCanvas.height = sendHeight;
    const ctx = modelCanvas.getContext('2d');

    const actualW = videoRef.current.videoWidth;
    const actualH = videoRef.current.videoHeight;

    modelCanvas.width = actualW;
    modelCanvas.height = actualH;
    ctx.drawImage(videoRef.current, 0, 0, actualW, actualH);

    modelCanvas.toBlob((blob) => {
      if (blob) socketRef.current.send(blob);
    }, 'image/png');
  };

  const drawOverlay = (detections) => {
    if (!canvasRef.current || !detections) return;
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    const scaleX = canvasRef.current.width / 1280;
    const scaleY = canvasRef.current.height / 720;

    detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const rX = x1 * scaleX;
      const rY = y1 * scaleY;
      const rW = (x2 - x1) * scaleX;
      const rH = (y2 - y1) * scaleY;

      ctx.strokeStyle = det.color || '#10B981';
      ctx.lineWidth = 4;
      ctx.strokeRect(rX, rY, rW, rH);

      const text = `${det.class} | ${det.condition} (${(det.confidence * 100).toFixed(0)}%)`;

      ctx.font = 'bold 20px monospace';
      const textMetrics = ctx.measureText(text);

      ctx.fillStyle = det.color || '#10B981';
      ctx.fillRect(rX, rY - 30, textMetrics.width + 10, 30);

      ctx.fillStyle = '#000000';
      ctx.fillText(text, rX + 5, rY - 8);
    });
  };

  const handleScan = async () => {
    const result = liveData?.best_result || (liveData?.confidence ? liveData : null);

    if (!result) {
      alert('No object detected yet.');
      return;
    }

    setIsScanning(true);

    const typeMap = {
      PET: 1,
      0: 1,
      HDPE: 2,
      1: 2,
      PVC: 3,
      2: 3,
      LDPE: 4,
      3: 4,
      PP: 5,
      4: 5,
      PS: 6,
      5: 6,
      Other: 7,
      6: 7,
    };

    const rawLabel = result.class || 'Other';
    const plasticTypeID = typeMap[rawLabel] || 7;

    const newScanData = {
      plasticTypeID: plasticTypeID,
      condition: result.condition || 'Clean',
      confidence: parseFloat(result.confidence || 0),
      status: 'New',
      timestamp: new Date().toISOString(),
    };

    try {
      await db.collection.add(newScanData);
      setIsScanning(false);
      alert(`Saved: ${rawLabel} (${result.condition})`);
      if (onScanComplete) onScanComplete();
    } catch (e) {
      console.error('Error saving scan', e);
      alert('Error saving to database: ' + e.message);
      setIsScanning(false);
    }
  };

  return (
    <div className="h-full flex flex-col relative bg-black">
      <div className="flex-1 relative overflow-hidden flex items-center justify-center">
        {!isConnected && (
          <div className="absolute top-4 left-4 z-50 bg-red-500/80 text-white px-3 py-1 rounded-full text-xs font-bold flex items-center gap-2">
            <Loader2 className="animate-spin" size={12} /> Connecting to AI...
          </div>
        )}

        <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover" />
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover pointer-events-none" />

        {isScanning && (
          <div className="absolute inset-0 z-20 pointer-events-none bg-emerald-500/20 animate-pulse">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="bg-slate-900/90 text-emerald-400 px-6 py-3 rounded-full font-mono border border-emerald-500/30 backdrop-blur-md">
                SAVING...
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="bg-white p-6 border-t border-slate-200 z-10 flex justify-center items-center gap-8">
        <button className="p-4 rounded-full bg-slate-100 text-slate-400 hover:bg-slate-200 transition-colors">
          <Settings size={24} />
        </button>

        <button
          onClick={handleScan}
          disabled={!isConnected || isScanning}
          className={`w-20 h-20 rounded-full border-4 border-slate-200 flex items-center justify-center shadow-lg transition-all transform hover:scale-105 active:scale-95 
            ${isConnected ? 'bg-emerald-600 hover:bg-emerald-500' : 'bg-slate-400 cursor-not-allowed'}`}
        >
          <Camera size={32} className="text-white" />
        </button>

        <button className="p-4 rounded-full bg-slate-100 text-slate-400 hover:bg-slate-200 transition-colors">
          <History size={24} />
        </button>
      </div>
    </div>
  );
};

export default Scanner;
