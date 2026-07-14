"""
Inference Module
================
Load a saved model and predict normal vs abnormal brain activity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

CLASS_NAMES = {0: "Normal", 1: "Abnormal"}


def load_model(model_path: PathLike = "saved_models/trained_model.pkl"):
    """Load a joblib-serialized sklearn pipeline."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first (python run_pipeline.py)."
        )
    logger.info("Loading model from %s", model_path)
    return joblib.load(model_path)


def load_meta(meta_path: PathLike = "saved_models/model_meta.json") -> Dict[str, Any]:
    """Load training metadata if available."""
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as fh:
        return json.load(fh)


def predict_from_features(
    model,
    features: np.ndarray,
    subject_id: str = "DEMO-001",
) -> Dict[str, Any]:
    """
    Predict class from a connectivity feature vector.

    Parameters
    ----------
    model : fitted sklearn pipeline
    features : 1D or 2D array
    subject_id : str

    Returns
    -------
    dict with prediction, confidence, class label, probabilities
    """
    X = np.atleast_2d(features)
    pred = int(model.predict(X)[0])
    label = CLASS_NAMES.get(pred, str(pred))

    confidence = None
    probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        probabilities = {CLASS_NAMES.get(i, str(i)): float(p) for i, p in enumerate(proba)}
        confidence = float(proba[pred])

    result = {
        "subject_id": subject_id,
        "prediction": pred,
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "n_features": int(X.shape[1]),
    }
    logger.info(
        "Prediction for %s: %s (confidence=%.3f)",
        subject_id,
        label,
        confidence if confidence is not None else -1,
    )
    return result


def predict_subject(
    features: np.ndarray,
    subject_id: str = "DEMO-001",
    model_path: PathLike = "saved_models/trained_model.pkl",
    model=None,
) -> Dict[str, Any]:
    """
    High-level inference: load model if needed and predict.
    """
    if model is None:
        model = load_model(model_path)
    meta = load_meta(Path(model_path).parent / "model_meta.json")
    result = predict_from_features(model, features, subject_id=subject_id)
    result["model_name"] = meta.get("best_model_name", "unknown")
    return result


def display_prediction(result: Dict[str, Any]) -> str:
    """Format prediction as a human-readable string."""
    lines = [
        "=" * 50,
        "fMRI ACTIVITY PREDICTION",
        "=" * 50,
        f"Subject ID   : {result.get('subject_id')}",
        f"Prediction   : {result.get('label')} (class={result.get('prediction')})",
        f"Confidence   : {result.get('confidence', 'N/A')}",
        f"Model        : {result.get('model_name', 'N/A')}",
    ]
    if result.get("probabilities"):
        lines.append("Probabilities:")
        for k, v in result["probabilities"].items():
            lines.append(f"  - {k}: {v:.4f}")
    lines.append("=" * 50)
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Demo with random features matching a trained model if present
    model_file = Path("saved_models/trained_model.pkl")
    if not model_file.exists():
        print("No trained model found. Run: python run_pipeline.py")
    else:
        model = load_model(model_file)
        # Infer feature dimension from model
        n_feat = model.named_steps["scaler"].n_features_in_
        rng = np.random.default_rng(7)
        feats = rng.normal(size=n_feat)
        res = predict_subject(feats, subject_id="SUB-DEMO")
        display_prediction(res)
