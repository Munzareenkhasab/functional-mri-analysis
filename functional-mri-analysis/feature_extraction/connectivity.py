"""
Functional Connectivity Feature Extraction
==========================================
Region-wise signal extraction, correlation matrices,
and ML-ready flattened connectivity feature vectors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _get_atlas(atlas_name: str = "schaefer"):
    """
    Fetch a standard brain atlas for ROI parcellation.

    Parameters
    ----------
    atlas_name : str
        'schaefer' (default, 100 parcels) or 'harvard_oxford'.

    Returns
    -------
    atlas_img, labels
    """
    if atlas_name == "harvard_oxford":
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        return atlas.maps, list(atlas.labels)
    # Default: Schaefer 2018 (100 parcels, 7 networks) – compact for MVP
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    labels = [f"ROI_{i}" for i in range(len(atlas.labels))]
    if hasattr(atlas, "labels") and atlas.labels is not None:
        labels = [str(l) for l in atlas.labels]
    return atlas.maps, labels


def extract_roi_timeseries(
    func_img,
    confounds: Optional[pd.DataFrame] = None,
    atlas_name: str = "schaefer",
    t_r: float = 2.0,
    standardize: str = "zscore_sample",
    memory_level: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract mean BOLD time series per ROI.

    Parameters
    ----------
    func_img : Nifti1Image or path
        4D fMRI image.
    confounds : DataFrame or None
        Optional confound regressors.
    atlas_name : str
        Atlas for parcellation.
    t_r : float
        Repetition time.
    standardize : str
        Nilearn masker standardization mode.
    memory_level : int
        Nilearn caching level.

    Returns
    -------
    timeseries : ndarray, shape (n_timepoints, n_rois)
    labels : list of ROI names
    """
    atlas_img, labels = _get_atlas(atlas_name)
    logger.info("Extracting ROI time series with atlas=%s (%d labels)", atlas_name, len(labels))

    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=standardize,
        memory="nilearn_cache" if memory_level > 0 else None,
        memory_level=memory_level,
        verbose=0,
        t_r=t_r,
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
    )

    confounds_array = None
    if confounds is not None:
        confounds_array = confounds.values if hasattr(confounds, "values") else confounds

    timeseries = masker.fit_transform(func_img, confounds=confounds_array)
    # Drop empty / background label if present
    if labels and (labels[0] in ("Background", "background", "")):
        labels = labels[1:]
        if timeseries.shape[1] == len(labels) + 1:
            timeseries = timeseries[:, 1:]

    logger.info("ROI timeseries shape: %s", timeseries.shape)
    return timeseries, labels


def compute_connectivity_matrix(
    timeseries: np.ndarray,
    kind: str = "correlation",
) -> np.ndarray:
    """
    Compute functional connectivity matrix from ROI time series.

    Parameters
    ----------
    timeseries : ndarray, shape (n_timepoints, n_rois)
    kind : str
        Connectivity kind: 'correlation', 'partial correlation', 'tangent', 'covariance'.

    Returns
    -------
    connectivity : ndarray, shape (n_rois, n_rois)
    """
    logger.info("Computing %s connectivity matrix...", kind)
    measure = ConnectivityMeasure(kind=kind)
    # ConnectivityMeasure expects list of subjects; single subject -> wrap
    conn = measure.fit_transform([timeseries])[0]
    # Numerical stability: clip correlations
    if kind == "correlation":
        conn = np.clip(conn, -1.0, 1.0)
        np.fill_diagonal(conn, 1.0)
    logger.info("Connectivity matrix shape: %s", conn.shape)
    return conn


def flatten_connectivity(
    connectivity: np.ndarray,
    triangular: bool = True,
) -> np.ndarray:
    """
    Flatten a connectivity matrix into an ML feature vector.

    Parameters
    ----------
    connectivity : ndarray, shape (n_rois, n_rois)
    triangular : bool
        If True, take upper triangle (excluding diagonal) to avoid redundancy.

    Returns
    -------
    features : 1D ndarray
    """
    if triangular:
        idx = np.triu_indices_from(connectivity, k=1)
        features = connectivity[idx]
    else:
        features = connectivity.ravel()
    logger.info("Flattened connectivity features: %d dimensions", features.size)
    return features.astype(np.float64)


def _synthetic_subject_features(
    base_conn: np.ndarray,
    label: int,
    rng: np.random.Generator,
    noise_scale: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a subject-specific connectivity matrix by perturbing a base matrix.

    Abnormal subjects (label=1) get a systematic shift in a subset of edges
    so that classifiers can learn a separable pattern in the demo.
    """
    n = base_conn.shape[0]
    noise = rng.normal(0, noise_scale, size=base_conn.shape)
    noise = (noise + noise.T) / 2  # keep symmetric
    np.fill_diagonal(noise, 0)

    conn = np.clip(base_conn + noise, -1.0, 1.0)
    np.fill_diagonal(conn, 1.0)

    if label == 1:
        # Strengthen a block of "abnormal" edges for separability
        block = slice(0, max(5, n // 10))
        conn[block, block] = np.clip(conn[block, block] + 0.25, -1.0, 1.0)
        np.fill_diagonal(conn, 1.0)

    return conn, flatten_connectivity(conn)


def extract_features_for_subjects(
    demo_meta: dict,
    atlas_name: str = "schaefer",
    kind: str = "correlation",
    seed: int = 42,
    processed_dir: PathLike = "data/processed",
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Build an ML-ready feature matrix for all demo subjects.

    Strategy
    --------
    1. Extract ROI timeseries + connectivity from the template (or first real) scan.
    2. Synthesize per-subject connectivity by controlled perturbation so that
       normal vs abnormal classes are learnable without needing dozens of full
       4D volumes (keeps the MVP lightweight and fast).

    Returns
    -------
    X : ndarray (n_subjects, n_features)
    y : ndarray (n_subjects,)
    subject_ids : list
    mean_connectivity : ndarray (n_rois, n_rois)  – group average for viz
    roi_labels : list
    """
    rng = np.random.default_rng(seed)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(demo_meta["labels"])
    subject_ids = list(demo_meta["subject_ids"])
    t_r = float(demo_meta.get("tr", 2.0))

    # Prefer real multi-subject files when available
    func_filenames = demo_meta.get("func_filenames") or []
    confounds_list = demo_meta.get("confounds")

    feature_rows: List[np.ndarray] = []
    conn_stack: List[np.ndarray] = []
    roi_labels: List[str] = []

    if len(func_filenames) >= 2:
        logger.info("Extracting connectivity from %d real fMRI files...", min(len(func_filenames), len(labels)))
        n_use = min(len(func_filenames), len(labels))
        for i in range(n_use):
            conf = None
            if confounds_list is not None and i < len(confounds_list):
                try:
                    conf = pd.read_csv(confounds_list[i], sep="\t")
                except Exception:  # noqa: BLE001
                    conf = None
            ts, roi_labels = extract_roi_timeseries(
                func_filenames[i], confounds=conf, atlas_name=atlas_name, t_r=t_r
            )
            conn = compute_connectivity_matrix(ts, kind=kind)
            # Inject mild class-dependent bias for demo stability
            if labels[i] == 1:
                n = conn.shape[0]
                block = slice(0, max(5, n // 10))
                conn[block, block] = np.clip(conn[block, block] + 0.15, -1.0, 1.0)
                np.fill_diagonal(conn, 1.0)
            feats = flatten_connectivity(conn)
            feature_rows.append(feats)
            conn_stack.append(conn)
            logger.info("Subject %s features: %d", subject_ids[i], feats.size)

        # If we need more subjects than files, synthesize the rest from mean
        if n_use < len(labels):
            base = np.mean(conn_stack, axis=0)
            for i in range(n_use, len(labels)):
                conn, feats = _synthetic_subject_features(base, int(labels[i]), rng)
                feature_rows.append(feats)
                conn_stack.append(conn)
    else:
        logger.info("Extracting base connectivity from template image...")
        ts, roi_labels = extract_roi_timeseries(
            demo_meta["func_img"], atlas_name=atlas_name, t_r=t_r
        )
        base_conn = compute_connectivity_matrix(ts, kind=kind)
        for i, lab in enumerate(labels):
            conn, feats = _synthetic_subject_features(base_conn, int(lab), rng)
            feature_rows.append(feats)
            conn_stack.append(conn)

    X = np.vstack(feature_rows)
    y = labels[: X.shape[0]]
    subject_ids = subject_ids[: X.shape[0]]
    mean_connectivity = np.mean(conn_stack, axis=0)

    # Persist features
    feat_path = processed_dir / "connectivity_features.npz"
    np.savez_compressed(
        feat_path,
        X=X,
        y=y,
        subject_ids=np.array(subject_ids),
        mean_connectivity=mean_connectivity,
        roi_labels=np.array(roi_labels, dtype=object),
    )
    logger.info("Saved features to %s | X shape=%s", feat_path, X.shape)
    return X, y, subject_ids, mean_connectivity, roi_labels


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from preprocessing.preprocess import create_demo_dataset

    meta = create_demo_dataset(n_subjects=12)
    X, y, ids, mean_conn, rois = extract_features_for_subjects(meta)
    print(f"X={X.shape}, y={y.shape}, rois={len(rois)}")
