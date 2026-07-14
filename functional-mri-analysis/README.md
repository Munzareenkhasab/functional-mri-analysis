# Functional MRI Analysis Platform

A lightweight **MVP** that demonstrates a complete fMRI analysis pipeline:

1. Load fMRI NIfTI files (`.nii` / `.nii.gz`)
2. Preprocess brain scans
3. Extract functional connectivity features
4. Train simple ML classifiers
5. Detect normal vs abnormal brain activity (demo dataset)
6. Generate PDF/HTML analysis reports
7. Visualize connectivity matrices and brain activity

> **Disclaimer:** This is an educational demo, **not** a medical device. Do not use for clinical diagnosis.

---

## Architecture

```
functional-mri-analysis/
├── data/
│   ├── raw/                 # Demo / downloaded NIfTI references
│   └── processed/           # Cleaned images + feature matrices
├── preprocessing/
│   └── preprocess.py        # Load, mask, smooth, temporal clean
├── feature_extraction/
│   └── connectivity.py      # ROI timeseries + correlation matrices
├── models/
│   └── train.py             # RF / SVM / Logistic Regression
├── inference/
│   └── predict.py           # Load model + classify
├── visualization/
│   └── visualize.py         # Slices, heatmaps, ROC, importance
├── reports/
│   └── generate_report.py   # PDF (ReportLab) + HTML reports
├── saved_models/            # trained_model.pkl, scaler.pkl
├── outputs/                 # Figures + generated_report.pdf
├── run_pipeline.py          # End-to-end entry point
├── requirements.txt
└── README.md
```

**Pipeline flow**

```
NIfTI fMRI → Preprocess → ROI Connectivity Features
        → Train (RF / SVM / LR) → Best Model
        → Inference → Visualizations → PDF/HTML Report
```

---

## Dataset

If no local NIfTI files are provided, the pipeline **automatically** uses Nilearn’s open **development fMRI** sample (`fetch_development_fmri`).  

Supported formats:

- NIfTI (`.nii`)
- Compressed NIfTI (`.nii.gz`)

Demo labels: `0 = Normal`, `1 = Abnormal` (synthesized for the classification MVP when clinical labels are unavailable).

---

## Installation

```bash
# Python 3.11 recommended
cd functional-mri-analysis
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Training (full pipeline)

```bash
python run_pipeline.py
# optional knobs
python run_pipeline.py --n-subjects 24 --seed 7 --test-size 0.25
```

This will:

| Step | Output |
|------|--------|
| Load demo fMRI | `data/raw/` |
| Preprocess | `data/processed/demo_cleaned.nii.gz` |
| Extract connectivity | `data/processed/connectivity_features.npz` |
| Train models | metrics in `saved_models/metrics.json` |
| Save best model | `saved_models/trained_model.pkl` |
| Save scaler | `saved_models/scaler.pkl` |
| Visualizations | `outputs/*.png` |
| Reports | `outputs/generated_report.pdf`, `.html` |

---

## Inference only

After training:

```bash
python -m inference.predict
```

Or programmatically:

```python
from inference.predict import load_model, predict_subject, display_prediction
import numpy as np

model = load_model("saved_models/trained_model.pkl")
n_feat = model.named_steps["scaler"].n_features_in_
features = np.random.randn(n_feat)  # replace with real connectivity vector
result = predict_subject(features, subject_id="SUB-042")
display_prediction(result)
```

---

## Modules

### Preprocessing (`preprocessing/preprocess.py`)

- Load NIfTI images  
- Spatial smoothing (optional FWHM)  
- Brain masking (EPI mask)  
- Temporal detrend / band-pass / z-score standardization  

### Feature extraction (`feature_extraction/connectivity.py`)

- Schaefer-100 (or Harvard-Oxford) ROI parcellation  
- Pearson correlation connectivity matrix  
- Upper-triangle flatten → ML feature vector  

### Models (`models/train.py`)

| Model | Notes |
|-------|--------|
| Random Forest | Feature importances for visualization |
| SVM (RBF) | Probability estimates enabled |
| Logistic Regression | Linear baseline |

Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC.

### Visualization (`visualization/visualize.py`)

- Brain orthogonal slices (mean BOLD)  
- Connectivity heatmap  
- ROI correlation submatrix  
- Confusion matrix & ROC curve  
- RF feature importance  
- Model comparison bar chart  

### Reports (`reports/generate_report.py`)

PDF/HTML include:

- Patient ID (demo)  
- Prediction + confidence  
- Connectivity heatmap  
- Summary statistics  
- Model used + evaluation metrics  
- Timestamp  

---

## Screenshots

> Place pipeline screenshots here after running `python run_pipeline.py`.

| Figure | Path |
|--------|------|
| Connectivity heatmap | `outputs/connectivity_heatmap.png` |
| Brain slices | `outputs/brain_slices.png` |
| Confusion matrix | `outputs/confusion_matrix.png` |
| ROC curve | `outputs/roc_curve.png` |
| Feature importance | `outputs/feature_importance.png` |
| Model comparison | `outputs/model_comparison.png` |
| PDF report | `outputs/generated_report.pdf` |

---

## Project checklist

- [x] Dataset loads (Nilearn demo / NIfTI)
- [x] NIfTI preprocessing works
- [x] Connectivity features extracted
- [x] Models train (RF, SVM, LR)
- [x] Metrics generated
- [x] Model saved (`trained_model.pkl`)
- [x] Prediction works
- [x] PDF report generated
- [x] Visualizations generated
- [x] README completed

---

## Future work

- Support multi-session real clinical cohorts (BIDS)  
- Confound regression with full nuisance models  
- Graph-theoretic connectivity metrics (strength, modularity)  
- Deep learning baselines (e.g., BrainNetCNN) on larger datasets  
- Interactive 3D surface rendering  
- Cross-site harmonization (ComBat)  
- Uncertainty quantification for predictions  

---

## License

Demo / educational use only. Dependencies retain their respective licenses (NiBabel, Nilearn, scikit-learn, ReportLab, etc.).
