interface Metric {
  label: string;
  value: string;
  hint?: string;
  accent?: string;
}

export function MetricCards({ metrics }: { metrics: Metric[] }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {metrics.map((m) => (
        <div
          key={m.label}
          className="rounded-xl border border-slate-200/80 bg-white p-4 shadow-sm"
        >
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            {m.label}
          </p>
          <p className={`mt-1 text-2xl font-bold tabular-nums ${m.accent ?? "text-slate-900"}`}>
            {m.value}
          </p>
          {m.hint && <p className="mt-0.5 text-xs text-slate-400">{m.hint}</p>}
        </div>
      ))}
    </div>
  );
}
