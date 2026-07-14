import { useCallback, useEffect, useMemo, useState } from "react";
import { BrainSlices } from "./components/BrainSlices";
import { ConfusionMatrix } from "./components/ConfusionMatrix";
import { ConnectivityHeatmap } from "./components/ConnectivityHeatmap";
import { FeatureImportance } from "./components/FeatureImportance";
import { MetricCards } from "./components/MetricCards";
import { ModelComparison } from "./components/ModelComparison";
import { PipelineStepper } from "./components/PipelineStepper";
import { ReportPreview } from "./components/ReportPreview";
import { RocCurve } from "./components/RocCurve";
import {
  DEMO_SUMMARY,
  SUBJECTS,
  type ClassLabel,
  type PipelineSummary,
} from "./data/demoResults";

type Tab = "overview" | "preprocess" | "features" | "models" | "inference" | "report";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "preprocess", label: "Preprocess" },
  { id: "features", label: "Connectivity" },
  { id: "models", label: "Models" },
  { id: "inference", label: "Inference" },
  { id: "report", label: "Report" },
];

const STEP_LOGS = [
  "Fetching Nilearn development fMRI sample…",
  "Loading NIfTI volume (4D BOLD)…",
  "Computing EPI brain mask…",
  "Spatial smoothing FWHM=6mm…",
  "Temporal detrend + band-pass (0.01–0.1 Hz)…",
  "Z-score standardization per voxel…",
  "Schaefer-100 atlas ROI extraction…",
  "Building Pearson correlation matrix…",
  "Flattening upper-triangle connectivity features…",
  "Training RandomForest (n_estimators=100)…",
  "Training SVM (RBF, probability=True)…",
  "Training LogisticRegression…",
  "Cross-validation & metric comparison…",
  "Selecting best model by F1…",
  "Running inference on held-out subject…",
  "Rendering connectivity heatmap…",
  "Writing trained_model.pkl & scaler.pkl…",
  "Generating PDF/HTML analysis report…",
  "Pipeline complete.",
];

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(-1);
  const [completed, setCompleted] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [summary, setSummary] = useState<PipelineSummary | null>(null);
  const [selectedSubject, setSelectedSubject] = useState(SUBJECTS[6].id);
  const [livePrediction, setLivePrediction] = useState<PipelineSummary["prediction"] | null>(
    null,
  );

  const best = useMemo(() => {
    const s = summary ?? DEMO_SUMMARY;
    return s.metrics.find((m) => m.model === s.best_model) ?? s.metrics[0];
  }, [summary]);

  const runPipeline = useCallback(async () => {
    if (running) return;
    setRunning(true);
    setSummary(null);
    setLivePrediction(null);
    setLogs([]);
    setStep(0);
    setCompleted(0);
    setTab("overview");

    for (let i = 0; i < STEP_LOGS.length; i++) {
      await new Promise((r) => setTimeout(r, 280 + Math.random() * 220));
      setLogs((prev) => [...prev, STEP_LOGS[i]]);
      // Map log index → pipeline step (0..5)
      const mapped = Math.min(5, Math.floor((i / STEP_LOGS.length) * 6));
      setStep(mapped);
      setCompleted(mapped);
    }

    const subject = SUBJECTS.find((s) => s.id === selectedSubject) ?? SUBJECTS[6];
    const isAbnormal = subject.label === "Abnormal";
    const conf = isAbnormal ? 0.82 + Math.random() * 0.14 : 0.78 + Math.random() * 0.16;
    const prediction = {
      subject_id: subject.id,
      prediction: isAbnormal ? 1 : 0,
      label: subject.label as ClassLabel,
      confidence: conf,
      probabilities: {
        Normal: isAbnormal ? 1 - conf : conf,
        Abnormal: isAbnormal ? conf : 1 - conf,
      } as Record<ClassLabel, number>,
      model_name: "RandomForest",
      n_features: 4950,
      true_label: isAbnormal ? 1 : 0,
    };

    const result: PipelineSummary = {
      ...DEMO_SUMMARY,
      patient_id: subject.id,
      prediction,
      timestamp: new Date().toISOString(),
      elapsed_seconds: 12 + Math.random() * 8,
    };

    setSummary(result);
    setLivePrediction(prediction);
    setCompleted(6);
    setStep(5);
    setRunning(false);
    setTab("inference");
  }, [running, selectedSubject]);

  // Auto-load demo results on first paint so the UI is never empty
  useEffect(() => {
    setSummary(DEMO_SUMMARY);
    setLivePrediction(DEMO_SUMMARY.prediction);
    setCompleted(6);
    setStep(5);
  }, []);

  const display = summary ?? DEMO_SUMMARY;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-slate-200/80 bg-white/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-sky-500 to-indigo-600 shadow-lg shadow-sky-200">
              <svg className="h-5 w-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 3c-1.5 2-4 3.5-4 7a4 4 0 108 0c0-3.5-2.5-5-4-7z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 14c-2 1-3 2.5-3 4.5S7 21 12 21s7-1 7-2.5S18 15 16 14" />
              </svg>
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-slate-900 sm:text-lg">
                fMRI Analysis Platform
              </h1>
              <p className="text-[11px] text-slate-500">
                Connectivity · ML Classification · Reporting
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <select
              value={selectedSubject}
              onChange={(e) => setSelectedSubject(e.target.value)}
              disabled={running}
              className="hidden rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-xs font-medium text-slate-700 shadow-sm sm:block"
            >
              {SUBJECTS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id} · {s.label}
                </option>
              ))}
            </select>
            <button
              onClick={runPipeline}
              disabled={running}
              className="inline-flex items-center gap-2 rounded-lg bg-sky-600 px-3 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-60 sm:px-4 sm:text-sm"
            >
              {running ? (
                <>
                  <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                  Running…
                </>
              ) : (
                <>
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 010 1.972l-11.54 6.347a1.125 1.125 0 01-1.667-.986V5.653z" />
                  </svg>
                  Run Pipeline
                </>
              )}
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="mx-auto max-w-7xl overflow-x-auto px-4 sm:px-6">
          <nav className="flex gap-1 pb-0">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`relative whitespace-nowrap px-3 py-2.5 text-xs font-semibold transition sm:text-sm ${
                  tab === t.id ? "text-sky-700" : "text-slate-500 hover:text-slate-800"
                }`}
              >
                {t.label}
                {tab === t.id && (
                  <span className="absolute inset-x-1 bottom-0 h-0.5 rounded-full bg-sky-600" />
                )}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:px-6">
        {/* Pipeline stepper */}
        <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
          <PipelineStepper active={Math.max(0, step)} completed={completed} />
        </section>

        {/* Disclaimer banner */}
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-2.5 text-xs text-amber-800">
          <strong>Educational demo only.</strong> Not a medical device — do not use for clinical
          diagnosis. Python package lives in{" "}
          <code className="rounded bg-amber-100 px-1">functional-mri-analysis/</code>.
        </div>

        {tab === "overview" && (
          <div className="space-y-6">
            <MetricCards
              metrics={[
                {
                  label: "Subjects",
                  value: String(display.n_subjects),
                  hint: display.dataset_source,
                },
                {
                  label: "ROIs",
                  value: String(display.n_rois),
                  hint: display.atlas.split("(")[0].trim(),
                },
                {
                  label: "Features",
                  value: display.n_features.toLocaleString(),
                  hint: "Upper-triangle edges",
                },
                {
                  label: "Best Model",
                  value: display.best_model,
                  hint: `F1 ${best.f1.toFixed(2)}`,
                  accent: "text-sky-700",
                },
              ]}
            />

            <div className="grid gap-6 lg:grid-cols-5">
              <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm lg:col-span-3">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-bold text-slate-800">Architecture</h2>
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-500">
                    Python 3.11 · NiBabel · Nilearn · sklearn
                  </span>
                </div>
                <div className="grid gap-2 sm:grid-cols-3">
                  {[
                    {
                      title: "Preprocessing",
                      items: ["Load NIfTI", "Brain mask", "Smooth 6mm", "Band-pass", "Z-score"],
                      color: "from-sky-500 to-cyan-500",
                    },
                    {
                      title: "Features",
                      items: ["Schaefer-100", "ROI signals", "Correlation FC", "Flatten edges"],
                      color: "from-violet-500 to-purple-500",
                    },
                    {
                      title: "ML + Report",
                      items: ["RF / SVM / LR", "Metrics + ROC", "Inference", "PDF + HTML"],
                      color: "from-emerald-500 to-teal-500",
                    },
                  ].map((card) => (
                    <div
                      key={card.title}
                      className="rounded-xl border border-slate-100 bg-slate-50 p-3"
                    >
                      <div
                        className={`mb-2 inline-flex rounded-md bg-gradient-to-r ${card.color} px-2 py-0.5 text-[11px] font-bold text-white`}
                      >
                        {card.title}
                      </div>
                      <ul className="space-y-1">
                        {card.items.map((it) => (
                          <li key={it} className="flex items-center gap-1.5 text-xs text-slate-600">
                            <span className="h-1 w-1 rounded-full bg-slate-400" />
                            {it}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
                <div className="rounded-xl bg-slate-900 p-3 font-mono text-[11px] leading-relaxed text-emerald-400">
                  <p className="text-slate-500"># end-to-end</p>
                  <p>$ cd functional-mri-analysis</p>
                  <p>$ pip install -r requirements.txt</p>
                  <p>$ python run_pipeline.py</p>
                  <p className="mt-1 text-slate-500">
                    # → saved_models/trained_model.pkl · outputs/generated_report.pdf
                  </p>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm lg:col-span-2">
                <h2 className="mb-3 text-sm font-bold text-slate-800">Pipeline Log</h2>
                <div className="h-64 overflow-y-auto rounded-lg bg-slate-950 p-3 font-mono text-[11px] leading-5 text-slate-300">
                  {logs.length === 0 ? (
                    <p className="text-slate-500">
                      Click <span className="text-sky-400">Run Pipeline</span> to simulate the full
                      workflow. Demo results are preloaded below.
                    </p>
                  ) : (
                    logs.map((line, i) => (
                      <div key={i} className="flex gap-2">
                        <span className="shrink-0 text-slate-600">
                          {String(i + 1).padStart(2, "0")}
                        </span>
                        <span
                          className={
                            line.includes("complete")
                              ? "text-emerald-400"
                              : line.includes("Training")
                                ? "text-amber-300"
                                : line.includes("inference")
                                  ? "text-sky-300"
                                  : ""
                          }
                        >
                          {line}
                        </span>
                      </div>
                    ))
                  )}
                  {running && (
                    <div className="mt-1 animate-pulse text-sky-400">▌ processing…</div>
                  )}
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-[10px]">
                  <span className="rounded-full bg-emerald-50 px-2 py-0.5 font-medium text-emerald-700">
                    Normal: {display.class_balance.normal}
                  </span>
                  <span className="rounded-full bg-rose-50 px-2 py-0.5 font-medium text-rose-700">
                    Abnormal: {display.class_balance.abnormal}
                  </span>
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 font-medium text-slate-600">
                    TR = {display.preprocessing.t_r}s
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {tab === "preprocess" && (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="text-sm font-bold text-slate-800">Brain Slice Visualization</h2>
              <p className="text-xs text-slate-500">
                Mean BOLD signal after masking & temporal cleaning (demo rendering).
              </p>
              <BrainSlices />
            </div>
            <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="text-sm font-bold text-slate-800">Preprocessing Parameters</h2>
              <table className="w-full text-sm">
                <tbody className="divide-y divide-slate-100">
                  {[
                    ["Smoothing FWHM", `${display.preprocessing.smoothing_fwhm} mm`],
                    ["Standardize", display.preprocessing.standardize ? "z-score" : "off"],
                    ["Detrend", display.preprocessing.detrend ? "linear" : "off"],
                    ["High-pass", `${display.preprocessing.high_pass} Hz`],
                    ["Low-pass", `${display.preprocessing.low_pass} Hz`],
                    ["Repetition time (TR)", `${display.preprocessing.t_r} s`],
                    [
                      "Masked shape",
                      `${display.preprocessing.masked_shape[0]} × ${display.preprocessing.masked_shape[1]} (time × voxels)`,
                    ],
                    ["Atlas", display.atlas],
                  ].map(([k, v]) => (
                    <tr key={String(k)}>
                      <td className="py-2.5 text-xs font-medium text-slate-500">{k}</td>
                      <td className="py-2.5 text-right text-xs font-semibold text-slate-800">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="rounded-lg border border-sky-100 bg-sky-50 p-3 text-xs text-sky-800">
                Implemented in{" "}
                <code className="rounded bg-sky-100 px-1">preprocessing/preprocess.py</code> using
                Nilearn <code>clean_img</code>, <code>smooth_img</code>, and{" "}
                <code>compute_epi_mask</code>.
              </div>
            </div>
          </div>
        )}

        {tab === "features" && (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="flex flex-col items-center rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="mb-1 self-start text-sm font-bold text-slate-800">
                Functional Connectivity Matrix
              </h2>
              <p className="mb-4 self-start text-xs text-slate-500">
                Pearson correlation between ROI timeseries (display subset 40×40).
              </p>
              <ConnectivityHeatmap n={40} size={340} seed={42} />
            </div>
            <div className="space-y-4">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="mb-3 text-sm font-bold text-slate-800">Summary Statistics</h2>
                <MetricCards
                  metrics={[
                    {
                      label: "Mean r",
                      value: display.summary_stats.mean_correlation.toFixed(3),
                    },
                    {
                      label: "Std r",
                      value: display.summary_stats.std_correlation.toFixed(3),
                    },
                    {
                      label: "Edges",
                      value: display.summary_stats.n_edges.toLocaleString(),
                    },
                    {
                      label: "+ Edges",
                      value: `${(display.summary_stats.frac_positive_edges * 100).toFixed(0)}%`,
                    },
                  ]}
                />
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="mb-2 text-sm font-bold text-slate-800">Feature Vector</h2>
                <p className="text-xs leading-relaxed text-slate-600">
                  Upper triangle of the {display.n_rois}×{display.n_rois} matrix (excluding
                  diagonal) yields{" "}
                  <strong>{display.n_features.toLocaleString()} features</strong> per subject —
                  ready for scikit-learn classifiers. Extraction lives in{" "}
                  <code className="rounded bg-slate-100 px-1">feature_extraction/connectivity.py</code>.
                </p>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {["correlation", "Schaefer-100", "NiftiLabelsMasker", "ConnectivityMeasure"].map(
                    (tag) => (
                      <span
                        key={tag}
                        className="rounded-full bg-violet-50 px-2 py-0.5 text-[10px] font-medium text-violet-700"
                      >
                        {tag}
                      </span>
                    ),
                  )}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="mb-3 text-sm font-bold text-slate-800">ROI Correlation (zoom)</h2>
                <ConnectivityHeatmap n={16} size={240} seed={99} title="First 16 ROIs" />
              </div>
            </div>
          </div>
        )}

        {tab === "models" && (
          <div className="space-y-6">
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-sm font-bold text-slate-800">Model Comparison</h2>
                <span className="rounded-full bg-emerald-50 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-700">
                  Best: {display.best_model}
                </span>
              </div>
              <ModelComparison metrics={display.metrics} />
            </div>

            <div className="grid gap-6 lg:grid-cols-3">
              {display.metrics.map((m) => (
                <div
                  key={m.model}
                  className={`rounded-2xl border p-5 shadow-sm ${
                    m.model === display.best_model
                      ? "border-sky-300 bg-sky-50/40 ring-1 ring-sky-200"
                      : "border-slate-200 bg-white"
                  }`}
                >
                  <h3 className="text-sm font-bold text-slate-800">{m.model}</h3>
                  <dl className="mt-3 space-y-1.5 text-xs">
                    {(
                      [
                        ["Accuracy", m.accuracy],
                        ["Precision", m.precision],
                        ["Recall", m.recall],
                        ["F1", m.f1],
                        ["ROC AUC", m.roc_auc],
                        ["CV F1", m.cv_f1_mean],
                      ] as const
                    ).map(([label, val]) => (
                      <div key={label} className="flex justify-between">
                        <dt className="text-slate-500">{label}</dt>
                        <dd className="font-mono font-semibold tabular-nums text-slate-800">
                          {val.toFixed(3)}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </div>
              ))}
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <div className="flex flex-col items-center rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="mb-3 self-start text-sm font-bold text-slate-800">
                  Confusion Matrix — {display.best_model}
                </h2>
                <ConfusionMatrix matrix={best.confusion_matrix} />
              </div>
              <div className="flex flex-col items-center rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="mb-3 self-start text-sm font-bold text-slate-800">ROC Curve</h2>
                <RocCurve auc={best.roc_auc} />
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="mb-3 text-sm font-bold text-slate-800">
                Feature Importance (Random Forest)
              </h2>
              <FeatureImportance topK={14} />
            </div>
          </div>
        )}

        {tab === "inference" && (
          <div className="grid gap-6 lg:grid-cols-5">
            <div className="space-y-4 lg:col-span-3">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="mb-4 text-sm font-bold text-slate-800">Prediction Result</h2>
                {livePrediction ? (
                  <div className="space-y-4">
                    <div className="flex flex-wrap items-center gap-3">
                      <span className="text-xs text-slate-500">Subject</span>
                      <span className="rounded-lg bg-slate-100 px-2.5 py-1 font-mono text-sm font-bold text-slate-800">
                        {livePrediction.subject_id}
                      </span>
                      <span
                        className={`rounded-full px-3 py-1 text-sm font-bold ${
                          livePrediction.label === "Abnormal"
                            ? "bg-rose-100 text-rose-700"
                            : "bg-emerald-100 text-emerald-700"
                        }`}
                      >
                        {livePrediction.label}
                      </span>
                    </div>

                    <div>
                      <div className="mb-1 flex justify-between text-xs">
                        <span className="text-slate-500">Confidence</span>
                        <span className="font-semibold tabular-nums text-slate-800">
                          {(livePrediction.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                        <div
                          className={`h-full rounded-full transition-all duration-700 ${
                            livePrediction.label === "Abnormal"
                              ? "bg-gradient-to-r from-rose-400 to-rose-600"
                              : "bg-gradient-to-r from-emerald-400 to-emerald-600"
                          }`}
                          style={{ width: `${livePrediction.confidence * 100}%` }}
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      {(Object.entries(livePrediction.probabilities) as [ClassLabel, number][]).map(
                        ([label, p]) => (
                          <div
                            key={label}
                            className="rounded-xl border border-slate-100 bg-slate-50 p-3"
                          >
                            <p className="text-[11px] font-medium text-slate-500">{label}</p>
                            <p className="text-xl font-bold tabular-nums text-slate-900">
                              {(p * 100).toFixed(1)}%
                            </p>
                            <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-200">
                              <div
                                className={`h-full ${
                                  label === "Abnormal" ? "bg-rose-500" : "bg-emerald-500"
                                }`}
                                style={{ width: `${p * 100}%` }}
                              />
                            </div>
                          </div>
                        ),
                      )}
                    </div>

                    <dl className="grid grid-cols-2 gap-2 text-xs">
                      <div className="rounded-lg bg-slate-50 px-3 py-2">
                        <dt className="text-slate-400">Model</dt>
                        <dd className="font-semibold text-slate-800">
                          {livePrediction.model_name}
                        </dd>
                      </div>
                      <div className="rounded-lg bg-slate-50 px-3 py-2">
                        <dt className="text-slate-400">Features</dt>
                        <dd className="font-semibold tabular-nums text-slate-800">
                          {livePrediction.n_features}
                        </dd>
                      </div>
                      <div className="rounded-lg bg-slate-50 px-3 py-2">
                        <dt className="text-slate-400">Saved model</dt>
                        <dd className="font-mono text-[11px] text-slate-700">
                          trained_model.pkl
                        </dd>
                      </div>
                      <div className="rounded-lg bg-slate-50 px-3 py-2">
                        <dt className="text-slate-400">Scaler</dt>
                        <dd className="font-mono text-[11px] text-slate-700">scaler.pkl</dd>
                      </div>
                    </dl>
                  </div>
                ) : (
                  <p className="text-sm text-slate-500">Run the pipeline to generate a prediction.</p>
                )}
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="mb-3 text-sm font-bold text-slate-800">Cohort Subjects</h2>
                <div className="grid max-h-56 grid-cols-2 gap-2 overflow-y-auto sm:grid-cols-4">
                  {SUBJECTS.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => setSelectedSubject(s.id)}
                      className={`rounded-lg border px-2 py-2 text-left text-xs transition ${
                        selectedSubject === s.id
                          ? "border-sky-400 bg-sky-50 ring-1 ring-sky-200"
                          : "border-slate-100 bg-slate-50 hover:border-slate-200"
                      }`}
                    >
                      <p className="font-mono font-semibold text-slate-800">{s.id}</p>
                      <p
                        className={
                          s.label === "Abnormal" ? "text-rose-600" : "text-emerald-600"
                        }
                      >
                        {s.label}
                      </p>
                    </button>
                  ))}
                </div>
                <p className="mt-2 text-[11px] text-slate-400">
                  Select a subject, then click Run Pipeline to re-simulate inference.
                </p>
              </div>
            </div>

            <div className="lg:col-span-2">
              <ReportPreview summary={display} />
            </div>
          </div>
        )}

        {tab === "report" && (
          <div className="grid gap-6 lg:grid-cols-2">
            <ReportPreview summary={display} />
            <div className="space-y-4">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="mb-3 text-sm font-bold text-slate-800">Artifacts Written</h2>
                <ul className="space-y-2 text-xs">
                  {[
                    ["saved_models/trained_model.pkl", "Best sklearn pipeline"],
                    ["saved_models/scaler.pkl", "StandardScaler (also inside pipeline)"],
                    ["saved_models/metrics.json", "Full evaluation metrics"],
                    ["outputs/generated_report.pdf", "PDF analysis report (ReportLab)"],
                    ["outputs/generated_report.html", "HTML twin of the report"],
                    ["outputs/connectivity_heatmap.png", "FC matrix figure"],
                    ["outputs/brain_slices.png", "Mean BOLD slices"],
                    ["outputs/confusion_matrix.png", "Confusion matrix"],
                    ["outputs/roc_curve.png", "ROC curve"],
                    ["outputs/feature_importance.png", "RF importances"],
                    ["outputs/pipeline_summary.json", "Machine-readable summary"],
                  ].map(([path, desc]) => (
                    <li
                      key={path}
                      className="flex items-start gap-2 rounded-lg border border-slate-100 bg-slate-50 px-3 py-2"
                    >
                      <span className="mt-0.5 text-emerald-500">✓</span>
                      <div>
                        <p className="font-mono font-semibold text-slate-800">{path}</p>
                        <p className="text-slate-500">{desc}</p>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="mb-2 text-sm font-bold text-slate-800">How to generate for real</h2>
                <pre className="overflow-x-auto rounded-xl bg-slate-900 p-4 text-[11px] leading-relaxed text-slate-200">
{`cd functional-mri-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --n-subjects 20

# outputs/
#   generated_report.pdf
#   connectivity_heatmap.png
# saved_models/
#   trained_model.pkl
#   scaler.pkl`}
                </pre>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-400">
        Functional MRI Analysis Platform · MVP Demo · NiBabel · Nilearn · scikit-learn · ReportLab
      </footer>
    </div>
  );
}
