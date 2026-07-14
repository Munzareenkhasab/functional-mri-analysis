import { useMemo } from "react";
import { generateConnectivityMatrix } from "../data/demoResults";

interface Props {
  size?: number;
  n?: number;
  seed?: number;
  title?: string;
}

function valueToColor(v: number): string {
  // RdBu-like diverging colormap centered at 0
  const t = (v + 1) / 2; // 0..1
  if (t < 0.5) {
    const u = t * 2;
    const r = Math.round(49 + u * (247 - 49));
    const g = Math.round(54 + u * (247 - 54));
    const b = Math.round(149 + u * (247 - 149));
    return `rgb(${r},${g},${b})`;
  }
  const u = (t - 0.5) * 2;
  const r = Math.round(247 + u * (165 - 247));
  const g = Math.round(247 + u * (0 - 247));
  const b = Math.round(247 + u * (38 - 247));
  return `rgb(${r},${g},${b})`;
}

export function ConnectivityHeatmap({ size = 320, n = 40, seed = 42, title }: Props) {
  const matrix = useMemo(() => generateConnectivityMatrix(n, seed), [n, seed]);
  const cell = size / n;

  return (
    <div className="flex flex-col items-center gap-2">
      {title && <p className="text-sm font-medium text-slate-600">{title}</p>}
      <svg width={size} height={size} className="rounded-lg shadow-inner ring-1 ring-slate-200">
        {matrix.map((row, i) =>
          row.map((v, j) => (
            <rect
              key={`${i}-${j}`}
              x={j * cell}
              y={i * cell}
              width={cell + 0.3}
              height={cell + 0.3}
              fill={valueToColor(v)}
            />
          )),
        )}
      </svg>
      <div className="flex w-full max-w-[320px] items-center gap-2 text-[10px] text-slate-500">
        <span>−1</span>
        <div
          className="h-2 flex-1 rounded-full"
          style={{
            background: "linear-gradient(to right, #313695, #f7f7f7, #a50026)",
          }}
        />
        <span>+1</span>
      </div>
    </div>
  );
}
