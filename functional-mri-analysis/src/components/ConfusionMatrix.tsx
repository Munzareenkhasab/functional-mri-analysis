export function ConfusionMatrix({
  matrix,
  labels = ["Normal", "Abnormal"],
}: {
  matrix: number[][];
  labels?: string[];
}) {
  const flat = matrix.flat();
  const max = Math.max(...flat, 1);

  return (
    <div className="inline-block">
      <div className="mb-1 text-center text-[10px] font-medium uppercase tracking-wide text-slate-400">
        Predicted →
      </div>
      <div className="flex">
        <div className="flex w-6 flex-col justify-center">
          <span
            className="origin-center -rotate-90 whitespace-nowrap text-[10px] font-medium uppercase tracking-wide text-slate-400"
            style={{ width: 80 }}
          >
            True
          </span>
        </div>
        <div>
          <div className="mb-1 grid grid-cols-2 gap-1 text-center text-[10px] text-slate-500">
            {labels.map((l) => (
              <span key={l}>{l}</span>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-1">
            {matrix.map((row, i) =>
              row.map((v, j) => {
                const intensity = v / max;
                return (
                  <div
                    key={`${i}-${j}`}
                    className="flex h-16 w-20 flex-col items-center justify-center rounded-lg text-sm font-bold"
                    style={{
                      background: `rgba(37, 99, 235, ${0.12 + intensity * 0.75})`,
                      color: intensity > 0.55 ? "white" : "#1e3a5f",
                    }}
                  >
                    {v}
                    <span className="text-[9px] font-normal opacity-70">
                      {i === 0 && j === 0
                        ? "TN"
                        : i === 0 && j === 1
                          ? "FP"
                          : i === 1 && j === 0
                            ? "FN"
                            : "TP"}
                    </span>
                  </div>
                );
              }),
            )}
          </div>
          <div className="mt-1 grid grid-cols-2 gap-1 text-center text-[10px] text-slate-500">
            {labels.map((l) => (
              <span key={l} className="opacity-0">
                {l}
              </span>
            ))}
          </div>
          <div className="-mt-4 grid grid-cols-2 gap-1 text-center text-[10px] font-medium text-slate-600">
            {labels.map((l) => (
              <span key={l}>{l}</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
