"""fMRI preprocessing package."""

from .preprocess import load_nifti, preprocess_fmri, create_demo_dataset

__all__ = ["load_nifti", "preprocess_fmri", "create_demo_dataset"]
