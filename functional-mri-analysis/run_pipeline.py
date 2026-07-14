#!/usr/bin/env python3
"""
Functional MRI Analysis Platform — End-to-End Pipeline
======================================================
Loads (or downloads) demo fMRI data, preprocesses, extracts functional
connectivity features, trains classifiers, runs inference, generates
visualizations, and writes PDF/HTML reports.

Usage
-----
    python run_pipeline.py
    python run_pipeline.py --n-subjects 24 --seed 7
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path when executed as a script
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_extraction.connectivity import extract_features_for_subjects
from inference.predict import display_prediction, predict_subject
from models.train import save_artifacts, train_models
from preprocessing.preprocess import create_demo_dataset, preprocess_fmri, save_processed
from reports.generate_report import generate_html_report, generate_pdf_report
from visualization.visualize import generate_all_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fMRI Analysis Platform MVP pipeline")
    parser.add_argument("--n-subjects", type=int, default=20, help="Number of demo subjects")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.25, help="Hold-out fraction")
    parser.add_argument(
        "--skip-preprocess-save",
        action="store_true",
        help="Skip writing cleaned NIfTI to disk (faster)",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Patient/subject ID for the inference report (default: first test subject)",
    )
    return parser.parse_args()


def connectivity_summary(conn: np.ndarray) -> dict:
    """Compute simple summary statistics for the connectivity matrix."""
    tri = conn[np.triu_indices_from(conn, k=1)]
    return {
        "n_rois": int(conn.shape[0]),
        "n_edges": int(tri.size),
        "mean_correlation": float(np.mean(tri)),
        "std_correlation": float(np.std(tri)),
        "min_correlation": float(np.min(tri)),
        "max_correlation": float(np.max(tri)),
        "median_correlation": float(np.median(tri)),
        "frac_positive_edges": float(np.mean(tri > 0)),
    }


def main() -> int:
    args = parse_args()
    t0 = time.time()

    data_raw = ROOT / "data" / "raw"
    data_proc = ROOT / "data" / "processed"
    models_dir = ROOT / "saved_models"
    outputs_dir = ROOT / "outputs"

    for d in (data_raw, data_proc, models_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1/7 — Load demo fMRI dataset")
    logger.info("=" * 60)
    demo = create_demo_dataset(
        n_subjects=args.n_subjects,
        output_dir=data_raw,
        seed=args.seed,
    )
    logger.info("Source: %s | subjects: %d", demo["source"], demo["n_subjects"])

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2/7 — Preprocess fMRI")
    logger.info("=" * 60)
    cleaned_img, mask_img, masked_data = preprocess_fmri(
        demo["func_img"],
        smoothing_fwhm=6.0,
        standardize=True,
        detrend=True,
        t_r=demo["tr"],
    )
    if not args.skip_preprocess_save:
        save_processed(cleaned_img, mask_img, output_dir=data_proc, prefix="demo")
    logger.info("Masked time series shape: %s", masked_data.shape)

    # ------------------------------------------------------------------
    # 3. Feature extraction
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3/7 — Extract functional connectivity features")
    logger.info("=" * 60)
    X, y, subject_ids, mean_conn, roi_labels = extract_features_for_subjects(
        demo,
        atlas_name="schaefer",
        kind="correlation",
        seed=args.seed,
        processed_dir=data_proc,
    )
    logger.info("Feature matrix X=%s | labels y=%s", X.shape, y.shape)
    summary_stats = connectivity_summary(mean_conn)

    # ------------------------------------------------------------------
    # 4. Train models
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4/7 — Train classifiers (RF, SVM, Logistic Regression)")
    logger.info("=" * 60)
    train_result = train_models(X, y, test_size=args.test_size, seed=args.seed)
    artifact_paths = save_artifacts(train_result, output_dir=models_dir)
    logger.info("Best model: %s", train_result["best_model_name"])
    logger.info("\n%s", train_result["comparison"].to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Inference on a held-out (or first) subject
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5/7 — Inference")
    logger.info("=" * 60)
    # Use first test sample for demo prediction
    X_test = train_result["X_test"]
    y_test = train_result["y_test"]
    sample_idx = 0
    sample_features = X_test[sample_idx]
    # Map back to a subject id when possible
    patient_id = args.patient_id or (subject_ids[0] if subject_ids else "SUB-001")
    prediction = predict_subject(
        sample_features,
        subject_id=patient_id,
        model_path=artifact_paths["model"],
        model=train_result["best_model"],
    )
    prediction["true_label"] = int(y_test[sample_idx])
    display_prediction(prediction)

    # ------------------------------------------------------------------
    # 6. Visualizations
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6/7 — Generate visualizations")
    logger.info("=" * 60)
    figure_paths = generate_all_figures(
        func_img=cleaned_img,
        connectivity=mean_conn,
        train_result=train_result,
        roi_labels=roi_labels,
        output_dir=outputs_dir,
    )

    # ------------------------------------------------------------------
    # 7. Reports
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 7/7 — Generate PDF & HTML reports")
    logger.info("=" * 60)
    best_name = train_result["best_model_name"]
    best_metrics = train_result["metrics"][best_name]

    pdf_path = generate_pdf_report(
        prediction=prediction,
        metrics=best_metrics,
        figure_paths=figure_paths,
        output_path=outputs_dir / "generated_report.pdf",
        patient_id=patient_id,
        summary_stats=summary_stats,
        model_name=best_name,
    )
    html_path = generate_html_report(
        prediction=prediction,
        metrics=best_metrics,
        figure_paths=figure_paths,
        output_path=outputs_dir / "generated_report.html",
        patient_id=patient_id,
        summary_stats=summary_stats,
        model_name=best_name,
    )

    # Also copy key names expected by checklist
    # trained_model.pkl and scaler.pkl already in saved_models/
    # generated_report.pdf in outputs/

    # Persist a machine-readable pipeline summary for the React dashboard
    pipeline_summary = {
        "patient_id": patient_id,
        "prediction": prediction,
        "best_model": best_name,
        "metrics": {
            name: {
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m.get("roc_auc"),
                "confusion_matrix": m["confusion_matrix"],
            }
            for name, m in train_result["metrics"].items()
        },
        "comparison": train_result["comparison"].to_dict(orient="records"),
        "summary_stats": summary_stats,
        "dataset_source": demo["source"],
        "n_subjects": int(demo["n_subjects"]),
        "n_features": int(X.shape[1]),
        "n_rois": int(mean_conn.shape[0]),
        "class_balance": {
            "normal": int(np.sum(y == 0)),
            "abnormal": int(np.sum(y == 1)),
        },
        "artifacts": {
            "model": str(artifact_paths["model"].relative_to(ROOT)),
            "scaler": str(artifact_paths["scaler"].relative_to(ROOT)),
            "pdf_report": str(pdf_path.relative_to(ROOT)),
            "html_report": str(html_path.relative_to(ROOT)),
            "figures": {k: (str(v.relative_to(ROOT)) if v else None) for k, v in figure_paths.items()},
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    summary_path = outputs_dir / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(pipeline_summary, fh, indent=2)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1f s", elapsed)
    logger.info("Model      : %s", artifact_paths["model"])
    logger.info("Scaler     : %s", artifact_paths["scaler"])
    logger.info("PDF report : %s", pdf_path)
    logger.info("HTML report: %s", html_path)
    logger.info("Summary    : %s", summary_path)
    logger.info("=" * 60)

    # Final checklist printout
    checklist = {
        "Dataset loads": True,
        "NIfTI preprocessing works": cleaned_img is not None and masked_data.size > 0,
        "Connectivity features extracted": X.ndim == 2 and X.shape[0] == len(y),
        "Models train": train_result["best_model"] is not None,
        "Metrics generated": best_metrics.get("accuracy") is not None,
        "Model saved": artifact_paths["model"].exists(),
        "Prediction works": prediction.get("label") in ("Normal", "Abnormal"),
        "PDF report generated": pdf_path.exists(),
        "Visualizations generated": (outputs_dir / "connectivity_heatmap.png").exists(),
        "README completed": (ROOT / "README.md").exists(),
    }
    print("\n=== FINAL CHECKLIST ===")
    all_ok = True
    for item, ok in checklist.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {item}")
        all_ok = all_ok and ok
    print("=======================\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
