"""
fMRI Preprocessing Module
=========================
Load NIfTI images, apply basic normalization, brain masking,
temporal standardization, and optional smoothing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from nilearn import datasets, image, masking
from nilearn.image import clean_img, smooth_img

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def load_nifti(filepath: PathLike):
    """
    Load a NIfTI (.nii / .nii.gz) file.

    Parameters
    ----------
    filepath : str or Path
        Path to the NIfTI image.

    Returns
    -------
    nibabel.Nifti1Image
        Loaded fMRI image.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")

    if filepath.suffix not in {".nii", ".gz"} and not str(filepath).endswith(".nii.gz"):
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Use .nii or .nii.gz")

    logger.info("Loading NIfTI: %s", filepath)
    img = image.load_img(str(filepath))
    logger.info("Shape: %s | Affine shape: %s", img.shape, img.affine.shape)
    return img


def preprocess_fmri(
    img,
    smoothing_fwhm: Optional[float] = 6.0,
    standardize: bool = True,
    detrend: bool = True,
    low_pass: Optional[float] = 0.1,
    high_pass: Optional[float] = 0.01,
    t_r: float = 2.0,
    mask_img=None,
) -> Tuple[object, object, np.ndarray]:
    """
    Preprocess an fMRI image.

    Steps
    -----
    1. Optional spatial smoothing
    2. Brain masking
    3. Temporal cleaning (detrend, high/low-pass, standardization)

    Parameters
    ----------
    img : Nifti1Image
        Input 4D fMRI image.
    smoothing_fwhm : float or None
        FWHM of Gaussian smoothing kernel (mm). None disables smoothing.
    standardize : bool
        Z-score each voxel time series.
    detrend : bool
        Linear detrend each voxel time series.
    low_pass, high_pass : float or None
        Temporal filter cutoffs (Hz).
    t_r : float
        Repetition time in seconds.
    mask_img : Nifti1Image or None
        Optional precomputed brain mask.

    Returns
    -------
    cleaned_img : Nifti1Image
        Preprocessed 4D image.
    mask_img : Nifti1Image
        Brain mask used.
    masked_data : ndarray, shape (n_timepoints, n_voxels)
        Masked and cleaned time series.
    """
    logger.info("Starting fMRI preprocessing...")

    # 1. Spatial smoothing
    if smoothing_fwhm is not None and smoothing_fwhm > 0:
        logger.info("Smoothing with FWHM=%.1f mm", smoothing_fwhm)
        img = smooth_img(img, fwhm=smoothing_fwhm)

    # 2. Brain mask
    if mask_img is None:
        logger.info("Computing brain mask from EPI...")
        mask_img = masking.compute_epi_mask(img)

    # 3. Temporal cleaning
    logger.info(
        "Temporal cleaning (detrend=%s, standardize=%s, low_pass=%s, high_pass=%s)",
        detrend,
        standardize,
        low_pass,
        high_pass,
    )
    cleaned_img = clean_img(
        img,
        detrend=detrend,
        standardize="zscore_sample" if standardize else False,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        mask_img=mask_img,
    )

    # 4. Extract masked time series
    masked_data = masking.apply_mask(cleaned_img, mask_img)
    logger.info(
        "Preprocessing complete. Masked data shape: %s (time x voxels)",
        masked_data.shape,
    )
    return cleaned_img, mask_img, masked_data


def create_demo_dataset(
    n_subjects: int = 20,
    output_dir: PathLike = "data/raw",
    seed: int = 42,
) -> dict:
    """
    Create a demo fMRI dataset using Nilearn's sample data and synthetic labels.

    Downloads a small open fMRI sample when local files are unavailable,
    then synthesizes subject-level feature proxies for the ML demo.

    Parameters
    ----------
    n_subjects : int
        Number of synthetic subjects to generate labels for.
    output_dir : str or Path
        Directory for any downloaded/cached raw data references.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {
          "func_img": Nifti1Image,
          "mask_img": Nifti1Image or None,
          "labels": ndarray of 0/1,
          "subject_ids": list[str],
          "source": str,
          "tr": float,
        }
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Nilearn development fMRI demo dataset...")
    try:
        # Small open dataset packaged with nilearn
        development = datasets.fetch_development_fmri(n_subjects=min(n_subjects, 30))
        func_filenames = development.func
        confounds = getattr(development, "confounds", None)
        source = "nilearn_development_fmri"
        # Use first subject as the template 4D image for connectivity demo
        func_img = image.load_img(func_filenames[0])
        tr = 2.0
        n_available = len(func_filenames)
        logger.info("Loaded development fMRI: %d subjects available", n_available)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch development fMRI (%s). Falling back to MNI template + noise.", exc)
        # Fallback: create a synthetic 4D volume from MNI template
        template = datasets.load_mni152_template(resolution=2)
        data = np.asarray(template.dataobj, dtype=np.float64)
        n_time = 50
        # Add temporal noise to create a fake BOLD series
        series = np.stack(
            [data + rng.normal(0, 0.05, size=data.shape) for _ in range(n_time)],
            axis=-1,
        )
        func_img = image.new_img_like(template, series)
        confounds = None
        source = "synthetic_mni_template"
        tr = 2.0
        n_available = n_subjects
        func_filenames = []

    # Demo labels: 0 = normal, 1 = abnormal
    # Slightly imbalanced for realism (~55% normal)
    labels = rng.choice([0, 1], size=n_subjects, p=[0.55, 0.45])
    subject_ids = [f"SUB-{i + 1:03d}" for i in range(n_subjects)]

    meta = {
        "func_img": func_img,
        "func_filenames": func_filenames if "func_filenames" in dir() else [],
        "confounds": confounds,
        "mask_img": None,
        "labels": labels,
        "subject_ids": subject_ids,
        "source": source,
        "tr": tr,
        "n_subjects": n_subjects,
    }

    # Persist a small metadata file
    meta_path = output_dir / "demo_dataset_meta.txt"
    with open(meta_path, "w", encoding="utf-8") as fh:
        fh.write(f"source={source}\n")
        fh.write(f"n_subjects={n_subjects}\n")
        fh.write(f"tr={tr}\n")
        fh.write(f"labels={labels.tolist()}\n")
        fh.write(f"subject_ids={subject_ids}\n")

    logger.info("Demo dataset ready (source=%s, n_subjects=%d)", source, n_subjects)
    return meta


def save_processed(
    cleaned_img,
    mask_img,
    output_dir: PathLike = "data/processed",
    prefix: str = "subject",
) -> Tuple[Path, Path]:
    """Save preprocessed image and mask to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_dir / f"{prefix}_cleaned.nii.gz"
    mask_path = output_dir / f"{prefix}_mask.nii.gz"

    cleaned_img.to_filename(str(cleaned_path))
    mask_img.to_filename(str(mask_path))
    logger.info("Saved processed images to %s", output_dir)
    return cleaned_path, mask_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo = create_demo_dataset(n_subjects=10)
    cleaned, mask, masked = preprocess_fmri(demo["func_img"], t_r=demo["tr"])
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Masked data shape: {masked.shape}")
