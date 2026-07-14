import { useMemo } from "react";
import { generateRocCurve } from "../data/demoResults";

export function RocCurve({ auc = 0.91 }: { auc?: number }) {
  const { fpr, tpr } = useMemo(() => generateRocCurve(3), []);
  const w = 260;
  const h = 220;
  const pad = 32;

  const toX = (v: number) => pad + v * (w - pad * 1.2);
  const toY = (v: number) => h - pad - v * (h - pad * 1.4);

  const path = fpr.map((f, i) => `${i === 0 ? "M" : "L"} ${toX(f)} ${toY(tpr[i])}`).join(" ");

  return (
    <svg width={w} height={h} className="overflow-visible">
      {/* axes */}
      <line x1={pad} y1={h - pad} x2={w - 10} y2={h - pad} stroke="#94a3b8" strokeWidth={1} />
      <line x1={pad} y1={h - pad} x2={pad} y2={12} stroke="#94a3b8" strokeWidth={1} />
      {/* chance line */}
      <line
        x1={toX(0)}
        y1={toY(0)}
        x2={toX(1)}
        y2={toY(1)}
        stroke="#cbd5e1"
        strokeDasharray="4 3"
      />
      {/* ROC */}
      <path d={path} fill="none" stroke="#2563eb" strokeWidth={2.5} />
      <path
        d={`${path} L ${toX(1)} ${toY(0)} L ${toX(0)} ${toY(0)} Z`}
        fill="rgba(37,99,235,0.12)"
      />
      <text x={w / 2} y={h - 8} textAnchor="middle" className="fill-slate-500" fontSize={10}>
        False Positive Rate
      </text>
      <text
        x={12}
        y={h / 2}
        textAnchor="middle"
        className="fill-slate-500"
        fontSize={10}
        transform={`rotate(-90 12 ${h / 2})`}
      >
        True Positive Rate
      </text>
      <text x={w - 70} y={40} className="fill-blue-700 font-semibold" fontSize={12}>
        AUC = {auc.toFixed(2)}
      </text>
    </svg>
  );
}
