"""
Model Training Module
=====================
Train Random Forest, SVM, and Logistic Regression classifiers
on functional connectivity features. Compare and persist the best model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

CLASS_NAMES = ["Normal", "Abnormal"]


def _build_pipelines(seed: int = 42) -> Dict[str, Pipeline]:
    """Create sklearn pipelines with shared scaling."""
    return {
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=8,
                        min_samples_leaf=2,
                        random_state=seed,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
    }


def evaluate_model(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Compute accuracy, precision, recall, F1, confusion matrix, and ROC metrics.
    """
    y_pred = model.predict(X_test)
    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, zero_division=0, output_dict=True
        ),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }

    # ROC / AUC when probability estimates are available
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            fpr, tpr, thresholds = roc_curve(y_test, proba)
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }
            metrics["y_proba"] = proba.tolist()
        except ValueError:
            metrics["roc_auc"] = None
            metrics["roc_curve"] = None
    else:
        metrics["roc_auc"] = None
        metrics["roc_curve"] = None

    logger.info(
        "%s | acc=%.3f prec=%.3f rec=%.3f f1=%.3f auc=%s",
        model_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        f"{metrics['roc_auc']:.3f}" if metrics.get("roc_auc") is not None else "N/A",
    )
    return metrics


def train_models(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    seed: int = 42,
    cv_folds: int = 3,
) -> Dict[str, Any]:
    """
    Train RF, SVM, and Logistic Regression; return comparison results.

    Returns
    -------
    dict with keys:
        models, metrics, best_model_name, best_model, X_train, X_test, y_train, y_test
    """
    logger.info("Training models on X=%s y=%s", X.shape, y.shape)

    # Ensure both classes appear in train/test when possible
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    pipelines = _build_pipelines(seed=seed)
    all_metrics: Dict[str, Dict[str, Any]] = {}
    trained: Dict[str, Pipeline] = {}

    for name, pipe in pipelines.items():
        logger.info("Fitting %s...", name)
        pipe.fit(X_train, y_train)
        trained[name] = pipe

        # Cross-validation on training set
        try:
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=min(cv_folds, len(y_train)), scoring="f1")
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
        except Exception as exc:  # noqa: BLE001
            logger.warning("CV failed for %s: %s", name, exc)
            cv_mean, cv_std = float("nan"), float("nan")

        metrics = evaluate_model(pipe, X_test, y_test, model_name=name)
        metrics["cv_f1_mean"] = cv_mean
        metrics["cv_f1_std"] = cv_std
        all_metrics[name] = metrics

    # Select best by F1 then accuracy
    best_name = max(
        all_metrics.keys(),
        key=lambda n: (all_metrics[n]["f1"], all_metrics[n]["accuracy"]),
    )
    logger.info("Best model: %s (F1=%.3f)", best_name, all_metrics[best_name]["f1"])

    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m.get("roc_auc"),
                "cv_f1_mean": m.get("cv_f1_mean"),
            }
            for name, m in all_metrics.items()
        ]
    ).sort_values("f1", ascending=False)

    return {
        "models": trained,
        "metrics": all_metrics,
        "comparison": comparison,
        "best_model_name": best_name,
        "best_model": trained[best_name],
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def extract_feature_importance(
    model: Pipeline,
    n_features: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Extract feature importances from Random Forest (if applicable)."""
    clf = model.named_steps.get("clf")
    if clf is None:
        return None
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_)
        if n_features is not None:
            return imp[:n_features]
        return imp
    if hasattr(clf, "coef_"):
        return np.abs(np.asarray(clf.coef_)).ravel()
    return None


def save_artifacts(
    result: Dict[str, Any],
    output_dir: PathLike = "saved_models",
) -> Dict[str, Path]:
    """
    Save best model, scaler (via pipeline), metrics JSON, and comparison CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "trained_model.pkl"
    scaler_path = output_dir / "scaler.pkl"
    metrics_path = output_dir / "metrics.json"
    comparison_path = output_dir / "model_comparison.csv"
    meta_path = output_dir / "model_meta.json"

    best = result["best_model"]
    joblib.dump(best, model_path)

    # Also save scaler separately for inference clarity
    scaler = best.named_steps.get("scaler")
    if scaler is not None:
        joblib.dump(scaler, scaler_path)

    # JSON-serializable metrics (drop large arrays if needed – keep them for report)
    serializable = {}
    for name, m in result["metrics"].items():
        serializable[name] = {
            k: v
            for k, v in m.items()
            if k not in ("y_true", "y_pred", "y_proba")  # keep report lean; full in pkl context
        }
        # Still keep confusion matrix & roc
        serializable[name]["confusion_matrix"] = m["confusion_matrix"]
        if m.get("roc_curve"):
            serializable[name]["roc_curve"] = m["roc_curve"]

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)

    result["comparison"].to_csv(comparison_path, index=False)

    meta = {
        "best_model_name": result["best_model_name"],
        "n_train": int(len(result["y_train"])),
        "n_test": int(len(result["y_test"])),
        "class_names": CLASS_NAMES,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    paths = {
        "model": model_path,
        "scaler": scaler_path,
        "metrics": metrics_path,
        "comparison": comparison_path,
        "meta": meta_path,
    }
    logger.info("Artifacts saved to %s", output_dir)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Smoke test with random data
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 50))
    y = rng.integers(0, 2, size=40)
    res = train_models(X, y)
    save_artifacts(res)
    print(res["comparison"])
