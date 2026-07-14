import type { PipelineSummary } from "../data/demoResults";

export function ReportPreview({ summary }: { summary: PipelineSummary }) {
  const { prediction, summary_stats: stats, best_model, metrics } = summary;
  const best = metrics.find((m) => m.model === best_model) ?? metrics[0];
  const conf = (prediction.confidence * 100).toFixed(1);
  const isAbnormal = prediction.label === "Abnormal";

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-200 bg-gradient-to-r from-slate-900 to-slate-800 px-5 py-4 text-white">
        <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-sky-300">
          Analysis Report
        </p>
        <h3 className="text-lg font-bold">Functional MRI Analysis Report</h3>
        <p className="text-xs text-slate-300">AI-Assisted Brain Activity Classification (Demo MVP)</p>
      </div>

      <div className="space-y-4 p-5 text-sm">
        <section>
          <h4 className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">
            1. Case Summary
          </h4>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
            <dt className="text-slate-400">Patient ID</dt>
            <dd className="font-semibold text-slate-800">{summary.patient_id}</dd>
            <dt className="text-slate-400">Timestamp</dt>
            <dd className="font-mono text-slate-700">
              {new Date(summary.timestamp).toLocaleString()}
            </dd>
            <dt className="text-slate-400">Prediction</dt>
            <dd>
              <span
                className={`inline-flex rounded-full px-2 py-0.5 text-xs font-bold ${
                  isAbnormal
                    ? "bg-rose-100 text-rose-700"
                    : "bg-emerald-100 text-emerald-700"
                }`}
              >
                {prediction.label}
              </span>
            </dd>
            <dt className="text-slate-400">Confidence</dt>
            <dd className="font-semibold tabular-nums text-slate-800">{conf}%</dd>
            <dt className="text-slate-400">Model Used</dt>
            <dd className="font-semibold text-slate-800">{best_model}</dd>
            <dt className="text-slate-400">Features</dt>
            <dd className="tabular-nums text-slate-700">{prediction.n_features}</dd>
          </dl>
        </section>

        <section>
          <h4 className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">
            2. Evaluation Metrics
          </h4>
          <div className="overflow-hidden rounded-lg border border-slate-100">
            <table className="w-full text-xs">
              <thead className="bg-slate-800 text-white">
                <tr>
                  <th className="px-3 py-1.5 text-left font-medium">Metric</th>
                  <th className="px-3 py-1.5 text-right font-medium">Value</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {[
                  ["Accuracy", best.accuracy],
                  ["Precision", best.precision],
                  ["Recall", best.recall],
                  ["F1 Score", best.f1],
                  ["ROC AUC", best.roc_auc],
                ].map(([label, val]) => (
                  <tr key={String(label)} className="bg-slate-50/50">
                    <td className="px-3 py-1.5 text-slate-600">{label}</td>
                    <td className="px-3 py-1.5 text-right font-mono font-semibold tabular-nums text-slate-800">
                      {(val as number).toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section>
          <h4 className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">
            3. Connectivity Summary
          </h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {[
              ["ROIs", stats.n_rois],
              ["Edges", stats.n_edges],
              ["Mean r", stats.mean_correlation.toFixed(3)],
              ["Std r", stats.std_correlation.toFixed(3)],
              ["Min r", stats.min_correlation.toFixed(3)],
              ["Max r", stats.max_correlation.toFixed(3)],
            ].map(([k, v]) => (
              <div key={String(k)} className="rounded-lg bg-teal-50 px-2.5 py-1.5">
                <span className="text-teal-600">{k}</span>
                <span className="ml-2 font-semibold tabular-nums text-teal-900">{v}</span>
              </div>
            ))}
          </div>
        </section>

        <p className="border-t border-slate-100 pt-3 text-[10px] leading-relaxed text-slate-400">
          DISCLAIMER: Demonstration MVP for educational purposes only. Not a medical device.
          Do not use for clinical diagnosis. Artifacts also written by Python pipeline to{" "}
          <code className="rounded bg-slate-100 px-1">outputs/generated_report.pdf</code>.
        </p>
      </div>
    </div>
  );
}
