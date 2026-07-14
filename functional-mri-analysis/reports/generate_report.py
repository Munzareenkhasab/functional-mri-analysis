"""
Report Generation Module
========================
Automatically generate PDF and HTML analysis reports containing
patient ID, prediction, confidence, connectivity heatmap,
summary statistics, model used, evaluation metrics, and timestamp.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Heading1"],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=6,
            textColor=colors.HexColor("#0f172a"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHead",
            parent=styles["Heading2"],
            fontSize=13,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#1e3a5f"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyText2",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="FooterNote",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.gray,
            alignment=TA_CENTER,
        )
    )
    return styles


def _safe_image(path: Optional[PathLike], width: float = 5.5 * inch) -> Optional[Image]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning("Image not found for report: %s", p)
        return None
    img = Image(str(p))
    # Maintain aspect ratio
    aspect = img.imageHeight / float(img.imageWidth) if img.imageWidth else 0.75
    img.drawWidth = width
    img.drawHeight = width * aspect
    return img


def generate_pdf_report(
    prediction: Dict[str, Any],
    metrics: Dict[str, Any],
    figure_paths: Dict[str, Optional[Path]],
    output_path: PathLike = "outputs/generated_report.pdf",
    patient_id: Optional[str] = None,
    summary_stats: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> Path:
    """
    Build a multi-section PDF analysis report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    patient_id = patient_id or prediction.get("subject_id", "DEMO-001")
    model_name = model_name or prediction.get("model_name", metrics.get("model_name", "N/A"))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    styles = _styles()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="fMRI Analysis Report",
        author="fMRI Analysis Platform",
    )

    story: List[Any] = []

    # Header
    story.append(Paragraph("Functional MRI Analysis Report", styles["ReportTitle"]))
    story.append(Paragraph("AI-Assisted Brain Activity Classification (Demo MVP)", styles["FooterNote"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1e3a5f")))
    story.append(Spacer(1, 10))

    # Patient / prediction summary table
    story.append(Paragraph("1. Case Summary", styles["SectionHead"]))
    conf = prediction.get("confidence")
    conf_str = f"{conf * 100:.1f}%" if isinstance(conf, (int, float)) else "N/A"
    summary_data = [
        ["Patient ID", str(patient_id)],
        ["Timestamp", timestamp],
        ["Prediction", str(prediction.get("label", "N/A"))],
        ["Class Index", str(prediction.get("prediction", "N/A"))],
        ["Confidence", conf_str],
        ["Model Used", str(model_name)],
        ["Feature Dimensions", str(prediction.get("n_features", "N/A"))],
    ]
    if prediction.get("probabilities"):
        for k, v in prediction["probabilities"].items():
            summary_data.append([f"P({k})", f"{v * 100:.1f}%"])

    table = Table(summary_data, colWidths=[2.2 * inch, 4.0 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eef6")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)

    # Evaluation metrics
    story.append(Paragraph("2. Model Evaluation Metrics", styles["SectionHead"]))
    metric_rows = [
        ["Metric", "Value"],
        ["Accuracy", f"{metrics.get('accuracy', 0):.4f}"],
        ["Precision", f"{metrics.get('precision', 0):.4f}"],
        ["Recall", f"{metrics.get('recall', 0):.4f}"],
        ["F1 Score", f"{metrics.get('f1', 0):.4f}"],
        [
            "ROC AUC",
            f"{metrics['roc_auc']:.4f}" if metrics.get("roc_auc") is not None else "N/A",
        ],
        [
            "CV F1 (mean±std)",
            (
                f"{metrics.get('cv_f1_mean', float('nan')):.3f} ± "
                f"{metrics.get('cv_f1_std', float('nan')):.3f}"
            ),
        ],
    ]
    mtable = Table(metric_rows, colWidths=[2.2 * inch, 4.0 * inch])
    mtable.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(mtable)

    # Confusion matrix as text
    cm = metrics.get("confusion_matrix")
    if cm:
        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                f"Confusion Matrix [[TN, FP], [FN, TP]]: <b>{cm}</b>",
                styles["BodyText2"],
            )
        )

    # Summary statistics
    story.append(Paragraph("3. Connectivity Summary Statistics", styles["SectionHead"]))
    if summary_stats:
        stats_rows = [["Statistic", "Value"]] + [[str(k), str(v)] for k, v in summary_stats.items()]
    else:
        stats_rows = [
            ["Statistic", "Value"],
            ["Note", "Summary stats not provided"],
        ]
    stable = Table(stats_rows, colWidths=[2.2 * inch, 4.0 * inch])
    stable.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(stable)

    # Figures
    story.append(Paragraph("4. Visualizations", styles["SectionHead"]))

    for key, caption in [
        ("connectivity", "Functional Connectivity Heatmap"),
        ("brain_slices", "Mean BOLD Brain Slices"),
        ("confusion", "Confusion Matrix"),
        ("roc", "ROC Curve"),
        ("feature_importance", "Feature Importance (Random Forest)"),
        ("model_comparison", "Model Comparison"),
        ("roi_correlation", "ROI Correlation Submatrix"),
    ]:
        img = _safe_image(figure_paths.get(key) if figure_paths else None, width=5.4 * inch)
        if img is not None:
            story.append(Paragraph(caption, styles["BodyText2"]))
            story.append(Spacer(1, 4))
            story.append(img)
            story.append(Spacer(1, 10))

    # Disclaimer
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.gray))
    story.append(
        Paragraph(
            "DISCLAIMER: This is a demonstration MVP for educational purposes only. "
            "It is NOT a medical device and must not be used for clinical diagnosis or treatment decisions.",
            styles["FooterNote"],
        )
    )
    story.append(Paragraph(f"Generated at {timestamp}", styles["FooterNote"]))

    doc.build(story)
    logger.info("PDF report written to %s", output_path)
    return output_path


def generate_html_report(
    prediction: Dict[str, Any],
    metrics: Dict[str, Any],
    figure_paths: Dict[str, Optional[Path]],
    output_path: PathLike = "outputs/generated_report.html",
    patient_id: Optional[str] = None,
    summary_stats: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> Path:
    """Generate a lightweight HTML twin of the PDF report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    patient_id = patient_id or prediction.get("subject_id", "DEMO-001")
    model_name = model_name or prediction.get("model_name", metrics.get("model_name", "N/A"))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    conf = prediction.get("confidence")
    conf_str = f"{conf * 100:.1f}%" if isinstance(conf, (int, float)) else "N/A"

    def img_tag(key: str, caption: str) -> str:
        p = figure_paths.get(key) if figure_paths else None
        if p is None or not Path(p).exists():
            return ""
        # Relative path from report location
        rel = Path(p).name
        return f'<figure><img src="{rel}" alt="{caption}" style="max-width:100%;border:1px solid #e2e8f0;border-radius:8px;"/><figcaption>{caption}</figcaption></figure>'

    stats_html = ""
    if summary_stats:
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary_stats.items())
        stats_html = f"<table>{rows}</table>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>fMRI Analysis Report — {patient_id}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; color: #0f172a; background: #f8fafc; }}
    h1 {{ color: #0f172a; border-bottom: 2px solid #1e3a5f; padding-bottom: .4rem; }}
    h2 {{ color: #1e3a5f; margin-top: 1.6rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: .8rem 0; background: white; }}
    th, td {{ border: 1px solid #cbd5e1; padding: .5rem .75rem; text-align: left; }}
    th {{ background: #1e3a5f; color: white; }}
    tr:nth-child(even) {{ background: #f1f5f9; }}
    .badge {{ display:inline-block; padding:.25rem .75rem; border-radius:999px; font-weight:600;
              background: {"#fee2e2" if prediction.get("label") == "Abnormal" else "#dcfce7"};
              color: {"#991b1b" if prediction.get("label") == "Abnormal" else "#166534"}; }}
    figure {{ margin: 1rem 0; }}
    figcaption {{ font-size: .85rem; color: #64748b; margin-top: .35rem; }}
    .disclaimer {{ font-size: .8rem; color: #64748b; margin-top: 2rem; border-top: 1px solid #cbd5e1; padding-top: 1rem; }}
  </style>
</head>
<body>
  <h1>Functional MRI Analysis Report</h1>
  <p>AI-Assisted Brain Activity Classification (Demo MVP)</p>

  <h2>1. Case Summary</h2>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    <tr><td>Patient ID</td><td>{patient_id}</td></tr>
    <tr><td>Timestamp</td><td>{timestamp}</td></tr>
    <tr><td>Prediction</td><td><span class="badge">{prediction.get("label", "N/A")}</span></td></tr>
    <tr><td>Confidence</td><td>{conf_str}</td></tr>
    <tr><td>Model Used</td><td>{model_name}</td></tr>
    <tr><td>Feature Dimensions</td><td>{prediction.get("n_features", "N/A")}</td></tr>
  </table>

  <h2>2. Evaluation Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Accuracy</td><td>{metrics.get("accuracy", 0):.4f}</td></tr>
    <tr><td>Precision</td><td>{metrics.get("precision", 0):.4f}</td></tr>
    <tr><td>Recall</td><td>{metrics.get("recall", 0):.4f}</td></tr>
    <tr><td>F1 Score</td><td>{metrics.get("f1", 0):.4f}</td></tr>
    <tr><td>ROC AUC</td><td>{f"{metrics['roc_auc']:.4f}" if metrics.get("roc_auc") is not None else "N/A"}</td></tr>
  </table>

  <h2>3. Connectivity Summary Statistics</h2>
  {stats_html or "<p>No summary statistics provided.</p>"}

  <h2>4. Visualizations</h2>
  {img_tag("connectivity", "Functional Connectivity Heatmap")}
  {img_tag("brain_slices", "Mean BOLD Brain Slices")}
  {img_tag("confusion", "Confusion Matrix")}
  {img_tag("roc", "ROC Curve")}
  {img_tag("feature_importance", "Feature Importance")}
  {img_tag("model_comparison", "Model Comparison")}

  <p class="disclaimer">
    DISCLAIMER: This is a demonstration MVP for educational purposes only.
    It is NOT a medical device and must not be used for clinical diagnosis.
    Generated at {timestamp}.
  </p>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written to %s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_pred = {
        "subject_id": "SUB-001",
        "prediction": 0,
        "label": "Normal",
        "confidence": 0.87,
        "probabilities": {"Normal": 0.87, "Abnormal": 0.13},
        "n_features": 4950,
        "model_name": "RandomForest",
    }
    demo_metrics = {
        "model_name": "RandomForest",
        "accuracy": 0.8,
        "precision": 0.75,
        "recall": 0.7,
        "f1": 0.72,
        "roc_auc": 0.85,
        "confusion_matrix": [[3, 1], [1, 2]],
        "cv_f1_mean": 0.7,
        "cv_f1_std": 0.05,
    }
    path = generate_pdf_report(demo_pred, demo_metrics, {}, summary_stats={"mean_r": 0.12})
    print("Wrote", path)
