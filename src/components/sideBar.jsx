import React from 'react';
import {
  BarChart3,
  ScanLine,
  Database,
  Info,
  Settings,
  XCircle
} from 'lucide-react';

const Sidebar = ({ activePage, setActivePage, isMobile, isOpen, setIsOpen }) => {
  const navItems = [
    { id: 'dashboard', icon: BarChart3, label: 'Dashboard' },
    { id: 'scanner', icon: ScanLine, label: 'Live Scanner' },
    { id: 'history', icon: Database, label: 'Data Logs' },
    { id: 'information', icon: Info, label: 'Information' },

  ];

  const sidebarClasses = isMobile
    ? `fixed inset-y-0 left-0 z-50 w-64 bg-slate-900 text-white transform transition-transform duration-300 ease-in-out ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      }`
    : `w-64 bg-slate-900 text-white flex-shrink-0 hidden md:block relative`; 

  return (
    <>
      {isMobile && isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-40" onClick={() => setIsOpen(false)}></div>
      )}
      <div className={sidebarClasses}>
        <div className="p-6 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2 font-bold text-xl tracking-wider">
            <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center text-slate-900">B</div>
            BEBAS
          </div>
          {isMobile && (
            <button onClick={() => setIsOpen(false)}>
              <XCircle />
            </button>
          )}
        </div>
        <nav className="p-4 space-y-2">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                setActivePage(item.id);
                setIsOpen(false);
              }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                activePage === item.id
                  ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-900/50'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <item.icon size={20} />
              <span className="font-medium">{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="absolute bottom-0 w-full p-6 border-t border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-slate-700 flex items-center justify-center text-slate-300 font-bold">
              OP
            </div>
            <div>
              <p className="text-sm font-medium text-white">Operator</p>
              <p className="text-xs text-slate-500">Recycle Center</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;