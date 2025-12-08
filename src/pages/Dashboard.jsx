import React from 'react';
import { Database, CheckCircle2, AlertTriangle, RefreshCw } from 'lucide-react';
import StatCard from '../components/StatCard';
import { PLASTIC_STYLE_MAP } from '../utils/constants';

const Dashboard = ({ history, onViewAll }) => {
  const totalScans = history.length;
  const cleanCount = history.filter((h) => h.condition === 'Clean').length;
  const dirtyCount = history.filter((h) => h.condition !== 'Clean').length;
  const purityRate = totalScans > 0 ? Math.round((cleanCount / totalScans) * 100) : 0;

  // Get most common plastic type
  const typeCounts = history.reduce((acc, curr) => {
    const code = curr.type?.code || 'Unknown';
    acc[code] = (acc[code] || 0) + 1;
    return acc;
  }, {});
  const topType = Object.keys(typeCounts).sort((a, b) => typeCounts[b] - typeCounts[a])[0] || 'N/A';

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Facility Overview</h2>
        <p className="text-slate-500">Real-time tracking of waste processing.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard title="Total Processed" value={totalScans} subtext="+12% from yesterday" icon={Database} color="bg-blue-500" />
        <StatCard title="Purity Rate" value={`${purityRate}%`} subtext="Target: >85%" icon={CheckCircle2} color="bg-emerald-500" />
        <StatCard title="Contaminated" value={dirtyCount} subtext="Requires manual wash" icon={AlertTriangle} color="bg-orange-500" />
        <StatCard title="Top Material" value={topType} subtext="Highest volume today" icon={RefreshCw} color="bg-purple-500" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Material Composition Chart */}
        <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
          <h3 className="font-bold text-lg mb-6">Material Composition</h3>
          <div className="space-y-4">
            {PLASTIC_STYLE_MAP.map((style) => {
              const count = history.filter((h) => h.type?.code === style.code).length;
              const percentage = totalScans > 0 ? (count / totalScans) * 100 : 0;
              if (percentage === 0) return null;

              return (
                <div key={style.code}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="font-medium text-slate-700">{style.code}</span>
                    <span className="text-slate-500">{Math.round(percentage)}%</span>
                  </div>
                  <div className="w-full bg-slate-100 rounded-full h-2.5">
                    <div className={`h-2.5 rounded-full ${style.bg}`} style={{ width: `${percentage}%` }}></div>
                  </div>
                </div>
              );
            })}
            {totalScans === 0 && <div className="text-center text-slate-400 py-8">No data available yet. Start scanning!</div>}
          </div>
        </div>

        {/* Recent Logs Preview */}
        <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-bold text-lg">Recent Activity</h3>
            <button 
              onClick={onViewAll}
              className="text-emerald-600 text-sm font-medium hover:underline"
            >
              View All
            </button>
          </div>
          <div className="space-y-4">
            {history.slice(0, 5).map((item) => (
              <div key={item.id} className="flex items-center justify-between p-3 hover:bg-slate-50 rounded-lg transition-colors border border-transparent hover:border-slate-100">
                <div className="flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full ${item.type?.bg || 'bg-gray-300'}`}></div>
                  <div>
                    <p className="font-bold text-slate-800 text-sm">{item.type?.code || 'Unknown'}</p>
                    <p className="text-xs text-slate-500">{item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : 'Just now'}</p>
                  </div>
                </div>
                <div className="text-right">
                  <span className={`text-xs font-bold px-2 py-1 rounded-full ${item.condition === 'Clean' ? 'bg-emerald-100 text-emerald-700' : 'bg-orange-100 text-orange-700'}`}>
                    {item.condition}
                  </span>
                </div>
              </div>
            ))}
            {totalScans === 0 && <div className="text-center text-slate-400 py-8">No recent scans.</div>}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;