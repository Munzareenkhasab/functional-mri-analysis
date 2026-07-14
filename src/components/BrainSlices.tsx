/** Stylized axial / coronal / sagittal brain slice mock for demo visualization */
export function BrainSlices() {
  const slices = [
    { label: "Axial", rotate: 0 },
    { label: "Coronal", rotate: 0 },
    { label: "Sagittal", rotate: 0 },
  ];

  return (
    <div className="grid grid-cols-3 gap-3">
      {slices.map((s, idx) => (
        <div key={s.label} className="flex flex-col items-center gap-2">
          <div className="relative flex h-36 w-full items-center justify-center overflow-hidden rounded-xl bg-slate-950 ring-1 ring-slate-800">
            <svg viewBox="0 0 120 120" className="h-32 w-32 opacity-90">
              <defs>
                <radialGradient id={`g${idx}`} cx="50%" cy="45%" r="50%">
                  <stop offset="0%" stopColor="#fbbf24" stopOpacity="0.9" />
                  <stop offset="40%" stopColor="#ef4444" stopOpacity="0.7" />
                  <stop offset="70%" stopColor="#3b82f6" stopOpacity="0.5" />
                  <stop offset="100%" stopColor="#0f172a" stopOpacity="0.2" />
                </radialGradient>
              </defs>
              {/* Skull outline */}
              <ellipse cx="60" cy="58" rx="42" ry="48" fill="#1e293b" stroke="#475569" strokeWidth="2" />
              {/* Brain mass */}
              <ellipse cx="60" cy="60" rx="34" ry="38" fill={`url(#g${idx})`} opacity="0.85" />
              {/* Hemisphere split */}
              <path d="M60 22 V98" stroke="#94a3b8" strokeWidth="0.8" opacity="0.5" />
              {/* Activity blobs */}
              <circle cx={40 + idx * 8} cy={50 + idx * 4} r={8 - idx} fill="#fde047" opacity="0.7" />
              <circle cx={72 - idx * 5} cy={68} r={6 + idx} fill="#f97316" opacity="0.55" />
              <circle cx={55} cy={40 + idx * 6} r={4} fill="#22d3ee" opacity="0.6" />
            </svg>
            <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/50 to-transparent p-1.5 text-center text-[10px] text-slate-300">
              mean BOLD
            </div>
          </div>
          <span className="text-xs font-medium text-slate-500">{s.label}</span>
        </div>
      ))}
    </div>
  );
}
