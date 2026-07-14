const STEPS = [
  { id: 0, title: "Load Data", desc: "NIfTI fMRI" },
  { id: 1, title: "Preprocess", desc: "Mask · Smooth · Clean" },
  { id: 2, title: "Features", desc: "Connectivity" },
  { id: 3, title: "Train", desc: "RF · SVM · LR" },
  { id: 4, title: "Predict", desc: "Classify" },
  { id: 5, title: "Report", desc: "PDF · HTML" },
];

export function PipelineStepper({
  active,
  completed,
}: {
  active: number;
  completed: number;
}) {
  return (
    <div className="w-full overflow-x-auto">
      <div className="flex min-w-[640px] items-center justify-between gap-1 px-1">
        {STEPS.map((step, i) => {
          const done = i < completed;
          const current = i === active;
          return (
            <div key={step.id} className="flex flex-1 items-center">
              <div className="flex flex-col items-center gap-1">
                <div
                  className={`flex h-9 w-9 items-center justify-center rounded-full text-sm font-bold transition-all ${
                    done
                      ? "bg-emerald-500 text-white shadow-md shadow-emerald-200"
                      : current
                        ? "bg-sky-600 text-white shadow-md shadow-sky-200 ring-4 ring-sky-100"
                        : "bg-slate-200 text-slate-500"
                  }`}
                >
                  {done ? (
                    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    i + 1
                  )}
                </div>
                <div className="text-center">
                  <p
                    className={`text-xs font-semibold ${
                      current ? "text-sky-700" : done ? "text-emerald-700" : "text-slate-500"
                    }`}
                  >
                    {step.title}
                  </p>
                  <p className="text-[10px] text-slate-400">{step.desc}</p>
                </div>
              </div>
              {i < STEPS.length - 1 && (
                <div
                  className={`mx-1 mb-6 h-0.5 flex-1 rounded ${
                    i < completed ? "bg-emerald-400" : "bg-slate-200"
                  }`}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
