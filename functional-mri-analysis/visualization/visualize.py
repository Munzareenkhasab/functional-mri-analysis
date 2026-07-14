"""
Visualization Module
====================
Brain slices, connectivity heatmaps, ROI matrices,
feature importance, confusion matrices, and ROC curves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for servers / CI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nilearn import plotting

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Shared style
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "RdBu_r"


def _ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_brain_slices(
    img,
    output_path: PathLike = "outputs/brain_slices.png",
    title: str = "fMRI Mean BOLD Signal",
    display_mode: str = "ortho",
) -> Path:
    """
    Visualize orthogonal brain slices from a 3D/4D NIfTI image.
    For 4D inputs, the temporal mean is shown.
    """
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)

    # Collapse 4D -> mean 3D
    data = img.get_fdata()
    if data.ndim == 4:
        from nilearn.image import mean_img

        img = mean_img(img)

    fig = plt.figure(figsize=(10, 4))
    plotting.plot_stat_map(
        img,
        display_mode=display_mode,
        title=title,
        colorbar=True,
        figure=fig,
        annotate=True,
        black_bg=False,
        cmap="cold_hot",
    )
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved brain slices -> %s", output_path)
    return output_path


def plot_connectivity_heatmap(
    connectivity: np.ndarray,
    output_path: PathLike = "outputs/connectivity_heatmap.png",
    title: str = "Functional Connectivity Matrix",
    roi_labels: Optional[List[str]] = None,
    max_labels: int = 20,
) -> Path:
    """Plot ROI-to-ROI correlation heatmap."""
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)

    n = connectivity.shape[0]
    fig, ax = plt.subplots(figsize=(9, 7))
    vmax = np.nanmax(np.abs(connectivity)) or 1.0
    sns.heatmap(
        connectivity,
        cmap=PALETTE,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        square=True,
        ax=ax,
        cbar_kws={"label": "Correlation", "shrink": 0.8},
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"ROI (n={n})")
    ax.set_ylabel(f"ROI (n={n})")

    # Optional sparse tick labels
    if roi_labels and len(roi_labels) == n and n <= max_labels:
        short = [str(l)[:18] for l in roi_labels]
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_xticklabels(short, rotation=90, fontsize=7)
        ax.set_yticklabels(short, rotation=0, fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved connectivity heatmap -> %s", output_path)
    return output_path


def plot_roi_correlation_matrix(
    connectivity: np.ndarray,
    output_path: PathLike = "outputs/roi_correlation.png",
    n_rois: int = 30,
) -> Path:
    """Plot a zoomed subset of the ROI correlation matrix for readability."""
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)

    sub = connectivity[:n_rois, :n_rois]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        sub,
        cmap=PALETTE,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"label": "r", "shrink": 0.8},
    )
    ax.set_title(f"ROI Correlation Matrix (first {n_rois} ROIs)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved ROI correlation -> %s", output_path)
    return output_path


def plot_confusion_matrix(
    cm: Union[List[List[int]], np.ndarray],
    output_path: PathLike = "outputs/confusion_matrix.png",
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
) -> Path:
    """Plot a confusion matrix heatmap."""
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    class_names = class_names or ["Normal", "Abnormal"]
    cm = np.asarray(cm)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved confusion matrix -> %s", output_path)
    return output_path


def plot_roc_curve(
    roc_data: Optional[Dict[str, List[float]]],
    auc: Optional[float],
    output_path: PathLike = "outputs/roc_curve.png",
    title: str = "ROC Curve",
) -> Optional[Path]:
    """Plot ROC curve if data is available."""
    if not roc_data or auc is None:
        logger.warning("ROC data unavailable; skipping plot.")
        return None

    output_path = Path(output_path)
    _ensure_dir(output_path.parent)

    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved ROC curve -> %s", output_path)
    return output_path


def plot_feature_importance(
    importances: Optional[np.ndarray],
    output_path: PathLike = "outputs/feature_importance.png",
    top_k: int = 25,
    title: str = "Top Connectivity Feature Importances (Random Forest)",
) -> Optional[Path]:
    """Bar chart of top-k feature importances."""
    if importances is None or len(importances) == 0:
        logger.warning("No feature importances available; skipping plot.")
        return None

    output_path = Path(output_path)
    _ensure_dir(output_path.parent)

    importances = np.asarray(importances)
    top_k = min(top_k, len(importances))
    idx = np.argsort(importances)[-top_k:][::-1]
    values = importances[idx]
    labels = [f"edge_{i}" for i in idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette("viridis", n_colors=top_k)
    ax.barh(range(top_k), values[::-1], color=colors[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved feature importance -> %s", output_path)
    return output_path


def plot_model_comparison(
    comparison_df,
    output_path: PathLike = "outputs/model_comparison.png",
) -> Path:
    """Grouped bar chart of model metrics."""
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)

    metrics = ["accuracy", "precision", "recall", "f1"]
    available = [m for m in metrics if m in comparison_df.columns]
    plot_df = comparison_df.set_index("model")[available]

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved model comparison -> %s", output_path)
    return output_path


def generate_all_figures(
    func_img,
    connectivity: np.ndarray,
    train_result: Dict[str, Any],
    roi_labels: Optional[List[str]] = None,
    output_dir: PathLike = "outputs",
) -> Dict[str, Optional[Path]]:
    """
    Generate the full visualization suite used by the report and dashboard.
    """
    output_dir = _ensure_dir(output_dir)
    paths: Dict[str, Optional[Path]] = {}

    paths["brain_slices"] = plot_brain_slices(
        func_img, output_path=output_dir / "brain_slices.png"
    )
    paths["connectivity"] = plot_connectivity_heatmap(
        connectivity,
        output_path=output_dir / "connectivity_heatmap.png",
        roi_labels=roi_labels,
    )
    paths["roi_correlation"] = plot_roi_correlation_matrix(
        connectivity, output_path=output_dir / "roi_correlation.png"
    )

    best_name = train_result["best_model_name"]
    best_metrics = train_result["metrics"][best_name]
    paths["confusion"] = plot_confusion_matrix(
        best_metrics["confusion_matrix"],
        output_path=output_dir / "confusion_matrix.png",
        title=f"Confusion Matrix — {best_name}",
    )
    paths["roc"] = plot_roc_curve(
        best_metrics.get("roc_curve"),
        best_metrics.get("roc_auc"),
        output_path=output_dir / "roc_curve.png",
        title=f"ROC Curve — {best_name}",
    )

    # Feature importance from RF if available
    from models.train import extract_feature_importance

    rf_model = train_result["models"].get("RandomForest")
    imp = extract_feature_importance(rf_model) if rf_model is not None else None
    paths["feature_importance"] = plot_feature_importance(
        imp, output_path=output_dir / "feature_importance.png"
    )
    paths["model_comparison"] = plot_model_comparison(
        train_result["comparison"], output_path=output_dir / "model_comparison.png"
    )

    logger.info("Generated %d figures in %s", sum(1 for v in paths.values() if v), output_dir)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    conn = rng.uniform(-1, 1, size=(40, 40))
    conn = (conn + conn.T) / 2
    np.fill_diagonal(conn, 1)
    plot_connectivity_heatmap(conn, output_path="outputs/demo_conn.png")
    print("Demo heatmap written.")
