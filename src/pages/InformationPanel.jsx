import React, { useState, useEffect } from 'react';
import { Camera } from 'lucide-react';
import { db } from '../db';
import { PLASTIC_STYLE_MAP } from '../utils/constants';

const EXTENSIONS = ['png', 'jpg', 'jpeg']; // The order we will try

const InformationPanel = ({ initialTypeId }) => {
  const [types, setTypes] = useState([]);
  const [selectedType, setSelectedType] = useState(null);
  const [loading, setLoading] = useState(true);
  
  // New State: Tracks which extension we are currently trying (0 = png, 1 = jpg, etc.)
  const [extIndex, setExtIndex] = useState(0);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const dbTypes = await db.plastictype.toArray();
        const mergedData = dbTypes.map((dbItem) => {
          const styleData = PLASTIC_STYLE_MAP[dbItem.RICNumber - 1] || PLASTIC_STYLE_MAP[6];
          return { ...dbItem, ...styleData };
        });
        setTypes(mergedData);
        
        if (initialTypeId) {
           const target = mergedData.find(t => t.plasticTypeID === initialTypeId);
           if (target) setSelectedType(target);
           else if (mergedData.length > 0) setSelectedType(mergedData[0]);
        } else if (mergedData.length > 0) {
           setSelectedType(mergedData[0]);
        }

      } catch (error) {
        console.error("Failed to fetch info types", error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [initialTypeId]);

  // Reset the extension trial counter whenever the user selects a different plastic type
  useEffect(() => {
    setExtIndex(0);
  }, [selectedType]);

  if (loading) return <div className="p-10 text-slate-400">Loading library...</div>;

  return (
    <div className="flex h-full bg-white">
      <div className="w-64 bg-slate-800 flex flex-col border-r border-slate-700">
        <div className="p-6 border-b border-slate-700">
          <h2 className="text-xl font-bold text-white">Types</h2>
        </div>
        <div className="flex-1 overflow-y-auto">
          {types.map((type) => (
            <button
              key={type.plasticTypeID}
              onClick={() => setSelectedType(type)}
              className={`w-full text-left px-6 py-4 border-b border-slate-700 transition-colors ${
                selectedType?.plasticTypeID === type.plasticTypeID ? 'bg-emerald-600 text-white' : 'text-slate-300 hover:bg-slate-700'
              }`}
            >
              <span className="font-bold tracking-wider">{type.code}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-8 md:p-12 bg-slate-50">
        {selectedType ? (
          <div className="max-w-3xl mx-auto bg-white p-8 rounded-3xl shadow-sm border border-slate-200">
            <div className="mb-6">
              <h1 className="text-3xl font-bold text-slate-900 mb-2">
                {selectedType.code} <span className="text-slate-400 font-light">{selectedType.name }</span>
              </h1>
              <div className="flex items-center gap-4 text-sm font-medium">
                <span className="text-slate-500">Resin Identification Code (RIC) = {selectedType.RICNumber}</span>
                <span className={`px-2 py-0.5 rounded-full ${
                  selectedType.difficulty === 'easy' ? 'bg-emerald-100 text-emerald-700' : 
                  selectedType.difficulty === 'medium' ? 'bg-yellow-100 text-yellow-700' : 
                  'bg-red-100 text-red-700'
                }`}>
                  {selectedType.difficulty.charAt(0).toUpperCase() + selectedType.difficulty.slice(1)} to Recycle
                </span>
              </div>
            </div>

            {/* --- SMART IMAGE SECTION --- */}
            <div className="w-full h-64 bg-slate-200 rounded-2xl mb-8 relative overflow-hidden group flex items-center justify-center">
               <img 
                  // Uses the current index to decide extension: 1.png, then 1.jpg, etc.
                  key={`${selectedType.plasticTypeID}-${extIndex}`} // Key ensures React re-renders fresh when index changes
                  src={`/images/${selectedType.RICNumber}.${EXTENSIONS[extIndex]}`}
                  alt={`${selectedType.name} example`}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    // This logic runs if the image fails to load
                    if (extIndex < EXTENSIONS.length - 1) {
                        // If we haven't tried all extensions yet, try the next one
                        setExtIndex(prev => prev + 1);
                    } else {
                        // If we ran out of extensions, hide image and show placeholder
                        e.target.style.display = 'none'; 
                        e.target.nextSibling.style.display = 'flex';
                    }
                  }}
               />
               
               {/* Fallback Placeholder */}
               <div className="hidden w-full h-full absolute top-0 left-0 bg-slate-200 items-center justify-center flex-col">
                  <div className="text-slate-400 font-handwriting text-2xl rotate-[-5deg]">Image not found</div>
                  <div className="mt-2 text-slate-300"><Camera size={32} /></div>
               </div>
            </div>
            {/* --------------------------- */}

            <div className="space-y-6">
              <div>
                <h3 className="font-bold text-slate-900 mb-2">Found in:</h3>
                <p className="text-slate-700 leading-relaxed bg-slate-50 p-4 rounded-xl border border-slate-100">{selectedType.foundin}</p>
              </div>
              <div>
                <h3 className="font-bold text-slate-900 mb-2">Description & Recycling:</h3>
                <p className="text-slate-600 leading-relaxed whitespace-pre-line">{selectedType.description}</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-slate-400">Select a plastic type to view details</div>
        )}
      </div>
    </div>
  );
};

export default InformationPanel;