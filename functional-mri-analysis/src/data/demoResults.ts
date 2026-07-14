/**
 * Demo pipeline results mirroring outputs from functional-mri-analysis/run_pipeline.py
 * Used by the interactive web dashboard when the Python backend is not running.
 */

export type ClassLabel = "Normal" | "Abnormal";

export interface ModelMetrics {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  cv_f1_mean: number;
  confusion_matrix: number[][];
}

export interface PredictionResult {
  subject_id: string;
  prediction: number;
  label: ClassLabel;
  confidence: number;
  probabilities: Record<ClassLabel, number>;
  model_name: string;
  n_features: number;
  true_label?: number;
}

export interface ConnectivityStats {
  n_rois: number;
  n_edges: number;
  mean_correlation: number;
  std_correlation: number;
  min_correlation: number;
  max_correlation: number;
  median_correlation: number;
  frac_positive_edges: number;
}

export interface PipelineSummary {
  patient_id: string;
  prediction: PredictionResult;
  best_model: string;
  metrics: ModelMetrics[];
  summary_stats: ConnectivityStats;
  dataset_source: string;
  n_subjects: number;
  n_features: number;
  n_rois: number;
  class_balance: { normal: number; abnormal: number };
  elapsed_seconds: number;
  timestamp: string;
  preprocessing: {
    smoothing_fwhm: number;
    standardize: boolean;
    detrend: boolean;
    low_pass: number;
    high_pass: number;
    t_r: number;
    masked_shape: [number, number];
  };
  atlas: string;
  pipeline_steps: string[];
}

/** Deterministic PRNG for reproducible demo matrices */
function mulberry32(seed: number) {
  return function next() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Generate a symmetric correlation-like matrix for heatmaps */
export function generateConnectivityMatrix(n = 40, seed = 42): number[][] {
  const rand = mulberry32(seed);
  const m: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    m[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      // Block-structure + noise to look like real FC
      const sameBlock = Math.floor(i / 8) === Math.floor(j / 8);
      const base = sameBlock ? 0.55 : 0.05;
      const v = Math.max(-0.95, Math.min(0.95, base + (rand() - 0.5) * 0.5));
      m[i][j] = v;
      m[j][i] = v;
    }
  }
  // Abnormal-style boost on first block (for visual interest)
  for (let i = 0; i < 5; i++) {
    for (let j = i + 1; j < 5; j++) {
      m[i][j] = Math.min(0.95, m[i][j] + 0.2);
      m[j][i] = m[i][j];
    }
  }
  return m;
}

export function generateFeatureImportances(n = 25, seed = 7): { name: string; value: number }[] {
  const rand = mulberry32(seed);
  const items = Array.from({ length: n }, () => ({
    name: `edge_${Math.floor(rand() * 4950)}`,
    value: rand(),
  }));
  items.sort((a, b) => b.value - a.value);
  const max = items[0]?.value || 1;
  return items.map((it) => ({ ...it, value: it.value / max }));
}

export function generateRocCurve(seed = 3): { fpr: number[]; tpr: number[]; auc: number } {
  const rand = mulberry32(seed);
  const fpr: number[] = [0];
  const tpr: number[] = [0];
  let t = 0;
  for (let i = 1; i < 20; i++) {
    fpr.push(i / 20);
    t = Math.min(1, t + 0.04 + rand() * 0.08);
    // Concave ROC-ish curve
    const ideal = 1 - Math.pow(1 - i / 20, 2.2);
    tpr.push(Math.min(1, ideal * 0.85 + t * 0.15));
  }
  fpr.push(1);
  tpr.push(1);
  return { fpr, tpr, auc: 0.91 };
}

export const DEMO_SUMMARY: PipelineSummary = {
  patient_id: "SUB-007",
  prediction: {
    subject_id: "SUB-007",
    prediction: 1,
    label: "Abnormal",
    confidence: 0.86,
    probabilities: { Normal: 0.14, Abnormal: 0.86 },
    model_name: "RandomForest",
    n_features: 4950,
    true_label: 1,
  },
  best_model: "RandomForest",
  metrics: [
    {
      model: "RandomForest",
      accuracy: 0.9,
      precision: 0.88,
      recall: 0.89,
      f1: 0.88,
      roc_auc: 0.91,
      cv_f1_mean: 0.84,
      confusion_matrix: [
        [3, 0],
        [1, 1],
      ],
    },
    {
      model: "SVM",
      accuracy: 0.8,
      precision: 0.75,
      recall: 0.8,
      f1: 0.77,
      roc_auc: 0.86,
      cv_f1_mean: 0.78,
      confusion_matrix: [
        [2, 1],
        [0, 2],
      ],
    },
    {
      model: "LogisticRegression",
      accuracy: 0.8,
      precision: 0.78,
      recall: 0.75,
      f1: 0.76,
      roc_auc: 0.84,
      cv_f1_mean: 0.75,
      confusion_matrix: [
        [2, 1],
        [0, 2],
      ],
    },
  ],
  summary_stats: {
    n_rois: 100,
    n_edges: 4950,
    mean_correlation: 0.142,
    std_correlation: 0.218,
    min_correlation: -0.61,
    max_correlation: 0.94,
    median_correlation: 0.118,
    frac_positive_edges: 0.72,
  },
  dataset_source: "nilearn_development_fmri",
  n_subjects: 20,
  n_features: 4950,
  n_rois: 100,
  class_balance: { normal: 11, abnormal: 9 },
  elapsed_seconds: 47.3,
  timestamp: new Date().toISOString(),
  preprocessing: {
    smoothing_fwhm: 6.0,
    standardize: true,
    detrend: true,
    low_pass: 0.1,
    high_pass: 0.01,
    t_r: 2.0,
    masked_shape: [168, 24546],
  },
  atlas: "Schaefer 2018 (100 ROIs, 7 networks)",
  pipeline_steps: [
    "Load NIfTI fMRI",
    "Preprocess (smooth, mask, clean)",
    "Extract ROI timeseries",
    "Compute connectivity matrix",
    "Flatten features",
    "Train RF / SVM / LR",
    "Evaluate & select best model",
    "Inference",
    "Generate report",
  ],
};

export const SUBJECTS = Array.from({ length: 20 }, (_, i) => {
  const id = `SUB-${String(i + 1).padStart(3, "0")}`;
  const abnormal = [2, 4, 7, 9, 11, 13, 15, 17, 19].includes(i + 1);
  return {
    id,
    label: (abnormal ? "Abnormal" : "Normal") as ClassLabel,
    age: 8 + ((i * 3) % 12),
    sessions: 1,
  };
});
