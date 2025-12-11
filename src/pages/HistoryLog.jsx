import React from 'react';
import { ChevronRight, CheckCircle2, RefreshCw } from 'lucide-react';

const HistoryLog = ({ history, onTypeClick }) => {
  return (
    <div className="p-6 max-w-7xl mx-auto h-screen flex flex-col">
      <div className="mb-6 flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Analysis Log</h2>
          <p className="text-slate-500">Full audit trail of all scanned items.</p>
        </div>
        
      </div>

      <div className="bg-white border border-slate-200 rounded-2xl shadow-sm flex-1 overflow-hidden flex flex-col">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">Timestamp</th>
                <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">Plastic Type</th>
                <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">Condition</th>
                {/* ADDED CONFIDENCE HEADER */}
                <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">Confidence</th>
                <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {history.length === 0 ? (
                <tr>
                  <td colSpan="5" className="px-6 py-12 text-center text-slate-400">
                    No records found in Database.
                  </td>
                </tr>
              ) : (
                history.map((item) => (
                  <tr key={item.collectionID || item.id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-6 py-4 text-sm text-slate-600 font-mono">
                      {item.timestamp ? new Date(item.timestamp).toLocaleString() : 'N/A'}
                    </td>

                    <td className="px-6 py-4">
                      <button
                        onClick={() => onTypeClick && onTypeClick(item.plasticTypeID)}
                        className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold tracking-wide border border-transparent transition-transform hover:scale-105 hover:shadow-sm cursor-pointer
                        ${item.type?.bgLight || 'bg-gray-100'} 
                        ${item.type?.color || 'text-gray-600'}`}
                        title="Click to view details"
                      >
                        <div className={`w-2 h-2 rounded-full ${item.type?.bg || 'bg-gray-500'}`}></div>
                        {item.type?.code || 'Unknown'}
                      </button>
                    </td>

                    <td className="px-6 py-4 text-sm text-slate-700">{item.condition}</td>

                    {/* ADDED CONFIDENCE DATA */}
                    <td className="px-6 py-4 text-sm text-slate-600 font-mono">
                      {item.confidence ? Math.round(item.confidence * 100) : 0}%
                    </td>

                    <td className="px-6 py-4">
                      {item.condition === 'Clean' ? (
                        <span className="text-emerald-600 flex items-center gap-1 text-xs font-bold">
                          <CheckCircle2 size={14} /> SORTED
                        </span>
                      ) : (
                        <span className="text-orange-600 flex items-center gap-1 text-xs font-bold">
                          <RefreshCw size={14} /> WASH REQ
                        </span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default HistoryLog;
