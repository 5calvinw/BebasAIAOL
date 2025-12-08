import React, { useState, useEffect, useRef } from 'react';
import { Camera, Settings, History, CheckCircle2, AlertTriangle } from 'lucide-react';
import { db } from '../db';
import { PLASTIC_STYLE_MAP, CONDITIONS } from '../utils/constants';

const Scanner = ({ onScanComplete }) => {
  const videoRef = useRef(null);
  const [isScanning, setIsScanning] = useState(false);
  const [lastResult, setLastResult] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
    }
  };

  const handleScan = () => {
    setIsScanning(true);
    setLastResult(null);

    // Simulate AI Processing Time
    setTimeout(async () => {
      const typeIndex = Math.floor(Math.random() * PLASTIC_STYLE_MAP.length);
      const plasticTypeID = typeIndex + 1;
      const conditionIndex = Math.floor(Math.random() * CONDITIONS.length);
      const confidence = (Math.random() * (0.99 - 0.75) + 0.75).toFixed(2);
      
      const newScanData = {
        plasticTypeID: plasticTypeID,
        condition: CONDITIONS[conditionIndex],
        confidence: parseFloat(confidence),
        status: "New",
        timestamp: new Date().toISOString() 
      };

      try {
        await db.collection.add(newScanData);
        const styleData = PLASTIC_STYLE_MAP[typeIndex];
        const visualResult = { type: { ...styleData, code: styleData.code, name: "Scanned Item" }, ...newScanData };

        setLastResult(visualResult);
        setIsScanning(false);
        if(onScanComplete) onScanComplete();
      } catch (e) {
        console.error("Error saving scan", e);
        setIsScanning(false);
      }
    }, 1500);
  };

  return (
    <div className="h-full flex flex-col relative bg-black">
      <div className="flex-1 relative overflow-hidden flex items-center justify-center">
        {cameraActive ? (
          <video ref={videoRef} autoPlay playsInline className="absolute inset-0 w-full h-full object-cover opacity-90" />
        ) : (
          <div className="text-slate-500 flex flex-col items-center">
            <Camera size={48} className="mb-4 opacity-50" />
            <p>Camera feed unavailable in this preview.</p>
            <p className="text-sm">(Using simulation mode)</p>
          </div>
        )}

        {isScanning && (
          <div className="absolute inset-0 z-20 pointer-events-none">
            <div className="w-full h-1 bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.8)] animate-scan-down absolute top-0"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="bg-slate-900/80 text-emerald-400 px-6 py-3 rounded-full font-mono animate-pulse border border-emerald-500/30 backdrop-blur-md">ANALYZING PLASTIC...</div>
            </div>
          </div>
        )}
      </div>

      {lastResult && (
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 z-30 animate-in fade-in zoom-in duration-300">
          <div className="bg-white/95 backdrop-blur-xl rounded-2xl p-6 shadow-2xl border-2 border-emerald-500">
            <div className="flex justify-between items-start mb-4">
              <h3 className={`text-3xl font-black ${lastResult.type.color}`}>{lastResult.type.code}</h3>
              <div className="bg-slate-100 px-2 py-1 rounded text-xs font-mono text-slate-500">{(lastResult.confidence * 100).toFixed(0)}% CONFIDENCE</div>
            </div>
            <div className="flex items-center gap-3 mb-6 bg-slate-50 p-3 rounded-lg border border-slate-200">
              <div className={`p-2 rounded-full ${lastResult.condition === 'Clean' ? 'bg-emerald-100 text-emerald-600' : 'bg-orange-100 text-orange-600'}`}>
                {lastResult.condition === 'Clean' ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
              </div>
              <div>
                <p className="text-xs text-slate-400 uppercase font-bold tracking-wider">Condition</p>
                <p className="text-slate-900 font-bold">{lastResult.condition}</p>
              </div>
            </div>
            <button onClick={() => setLastResult(null)} className="w-full bg-slate-900 text-white py-3 rounded-xl font-bold hover:bg-slate-800 transition-colors">Scan Next Item</button>
          </div>
        </div>
      )}

      <div className="bg-white p-6 border-t border-slate-200 z-10 flex justify-center items-center gap-8">
        <button className="p-4 rounded-full bg-slate-100 text-slate-400 hover:bg-slate-200 hover:text-slate-600 transition-colors"><Settings size={24} /></button>
        <button onClick={handleScan} disabled={isScanning || lastResult} className={`w-20 h-20 rounded-full border-4 border-slate-200 flex items-center justify-center shadow-lg transition-all transform hover:scale-105 active:scale-95 ${isScanning ? 'bg-emerald-500 animate-pulse' : 'bg-emerald-600 hover:bg-emerald-500'}`}>
          <Camera size={32} className="text-white" />
        </button>
        <button className="p-4 rounded-full bg-slate-100 text-slate-400 hover:bg-slate-200 hover:text-slate-600 transition-colors"><History size={24} /></button>
      </div>
    </div>
  );
};

export default Scanner;