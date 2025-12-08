import React, { useState, useEffect } from 'react';
import { Menu } from 'lucide-react';
import { db, seedCollections, seedPlasticTypes } from "./db";
import { PLASTIC_STYLE_MAP } from './utils/constants';

// Import Components
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Scanner from './pages/Scanner';
import HistoryLog from './pages/HistoryLog';
import InformationPanel from './pages/InformationPanel';

const App = () => {
  const [activePage, setActivePage] = useState('dashboard');
  const [history, setHistory] = useState([]);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  
  // New State: Tracks which plastic type to show in the Info panel
  const [selectedInfoId, setSelectedInfoId] = useState(null);

  useEffect(() => {
    const initDB = async () => {
       await seedPlasticTypes();
       await seedCollections();
       loadData();
    };
    initDB();

    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const loadData = async () => {
    try {
      const allCollections = await db.collection.toArray();
      const allTypesDB = await db.plastictype.toArray();

      const formattedHistory = allCollections.map(item => {
        const dbTypeInfo = allTypesDB.find(t => t.plasticTypeID === item.plasticTypeID);
        const ricIndex = dbTypeInfo ? dbTypeInfo.RICNumber - 1 : 6;
        const styleInfo = PLASTIC_STYLE_MAP[ricIndex] || PLASTIC_STYLE_MAP[6];

        const fullTypeInfo = {
          code: styleInfo.code, 
          name: dbTypeInfo ? dbTypeInfo.code : 'Unknown', 
          ...styleInfo
        };

        return { ...item, type: fullTypeInfo, id: item.collectionID };
      });

      const sortedHistory = formattedHistory.sort((a, b) => {
         const timeA = a.timestamp ? new Date(a.timestamp) : 0;
         const timeB = b.timestamp ? new Date(b.timestamp) : 0;
         return timeB - timeA;
      });

      setHistory(sortedHistory);
    } catch (error) {
      console.error("Error loading history from DB:", error);
    }
  };

  const handleScanComplete = () => {
    loadData();
  };

  // --- NEW HANDLER ---
  const handleTypeClick = (plasticTypeID) => {
    setSelectedInfoId(plasticTypeID); // Set the specific ID
    setActivePage('information');     // Switch tabs
  };

  return (
    <div className="flex h-screen bg-slate-50 font-sans text-slate-900 overflow-hidden">
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
        isMobile={isMobile}
        isOpen={isMobileMenuOpen}
        setIsOpen={setIsMobileMenuOpen}
      />

      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {isMobile && (
          <header className="bg-white border-b border-slate-200 p-4 flex items-center justify-between z-10">
            <div className="flex items-center gap-2 font-bold text-lg">
              <div className="w-6 h-6 bg-emerald-500 rounded flex items-center justify-center text-slate-900 text-xs">B</div>
              BEBAS
            </div>
            <button onClick={() => setIsMobileMenuOpen(true)}>
              <Menu className="text-slate-600" />
            </button>
          </header>
        )}

        <div className="flex-1 overflow-y-auto h-full">
          {activePage === 'dashboard' && (
            <Dashboard 
              history={history} 
              onViewAll={() => setActivePage('history')} 
            />
          )}
          {activePage === 'scanner' && <Scanner onScanComplete={handleScanComplete} />}
          
          {/* Pass the click handler to HistoryLog */}
          {activePage === 'history' && (
            <HistoryLog 
              history={history} 
              onTypeClick={handleTypeClick} 
            />
          )}
          
          {/* Pass the selected ID to InformationPanel */}
          {activePage === 'information' && (
            <InformationPanel initialTypeId={selectedInfoId} />
          )}
          
          {activePage === 'settings' && (
            <div className="flex items-center justify-center h-full text-slate-400">Settings Panel Placeholder</div>
          )}
        </div>
      </main>

      <style>{`
        @keyframes scan-down {
          0% { top: 0; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        .animate-scan-down {
          animation: scan-down 2s linear infinite;
        }
      `}</style>
    </div>
  );
};

export default App;