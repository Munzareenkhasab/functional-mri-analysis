"""Inference package."""

from .predict import load_model, predict_subject, predict_from_features

__all__ = ["load_model", "predict_subject", "predict_from_features"]
