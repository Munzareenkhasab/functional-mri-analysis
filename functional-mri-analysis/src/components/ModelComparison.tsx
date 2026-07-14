import type { ModelMetrics } from "../data/demoResults";

const METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"] as const;
const COLORS = ["#0ea5e9", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"];

export function ModelComparison({ metrics }: { metrics: ModelMetrics[] }) {
  const maxH = 140;

  return (
    <div className="overflow-x-auto">
      <div className="flex min-w-[420px] items-end justify-around gap-6 px-2 pb-2 pt-4">
        {metrics.map((m) => (
          <div key={m.model} className="flex flex-1 flex-col items-center gap-2">
            <div className="flex h-[150px] items-end gap-1">
              {METRIC_KEYS.map((k, i) => {
                const v = m[k];
                return (
                  <div key={k} className="group relative flex flex-col items-center">
                    <div
                      className="w-3 rounded-t-sm transition-all"
                      style={{
                        height: Math.max(4, v * maxH),
                        backgroundColor: COLORS[i],
                      }}
                      title={`${k}: ${v.toFixed(2)}`}
                    />
                  </div>
                );
              })}
            </div>
            <p className="text-center text-xs font-semibold text-slate-700">{m.model}</p>
            <p className="text-[10px] text-slate-400">F1 {m.f1.toFixed(2)}</p>
          </div>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap justify-center gap-3">
        {METRIC_KEYS.map((k, i) => (
          <div key={k} className="flex items-center gap-1.5 text-[10px] text-slate-500">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ background: COLORS[i] }} />
            {k.replace("_", " ")}
          </div>
        ))}
      </div>
    </div>
  );
}
