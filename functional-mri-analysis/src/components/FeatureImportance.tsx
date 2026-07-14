import { useMemo } from "react";
import { generateFeatureImportances } from "../data/demoResults";

export function FeatureImportance({ topK = 12 }: { topK?: number }) {
  const items = useMemo(() => generateFeatureImportances(topK, 11), [topK]);

  return (
    <div className="space-y-1.5">
      {items.map((it, i) => (
        <div key={it.name} className="flex items-center gap-2">
          <span className="w-16 shrink-0 truncate text-right text-[10px] text-slate-500">
            {it.name}
          </span>
          <div className="h-3 flex-1 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{
                width: `${it.value * 100}%`,
                background: `linear-gradient(90deg, #0ea5e9, #6366f1)`,
                opacity: 1 - i * 0.04,
              }}
            />
          </div>
          <span className="w-8 text-[10px] tabular-nums text-slate-400">
            {it.value.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}
