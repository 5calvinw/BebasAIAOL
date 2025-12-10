import React from 'react';

// We explicitly map the input colors to the specific Tailwind classes we want.
// This ensures Tailwind detects the classes during the build process.
const COLOR_VARIANTS = {
  'bg-blue-500': {
    bg: 'bg-blue-100',
    text: 'text-blue-600'
  },
  'bg-emerald-500': {
    bg: 'bg-emerald-100',
    text: 'text-emerald-600'
  },
  'bg-orange-500': {
    bg: 'bg-orange-100',
    text: 'text-orange-600'
  },
  'bg-purple-500': {
    bg: 'bg-purple-100',
    text: 'text-purple-600'
  }
};

const StatCard = ({ title, value, subtext, icon: Icon, color }) => {
  // Fallback to blue if a color is passed that isn't in our map
  const variant = COLOR_VARIANTS[color] || COLOR_VARIANTS['bg-blue-500'];

  return (
    <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-4">
        <div>
          <p className="text-slate-500 text-sm font-medium mb-1">{title}</p>
          <h3 className="text-2xl font-bold text-slate-900">{value}</h3>
        </div>
        
        {/* We use the explicit classes from our variant map */}
        <div className={`p-3 rounded-xl ${variant.bg}`}>
          <Icon size={24} className={variant.text} />
        </div>
      </div>
      <p className="text-xs text-slate-400 font-medium">{subtext}</p>
    </div>
  );
};

export default StatCard;