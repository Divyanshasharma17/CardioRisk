"""
report.py
---------
PDF report generation and risk factor breakdown using
logistic regression coefficients.
"""

import io
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Human-readable feature labels
FEATURE_LABELS = {
    "age":         "Age",
    "gender":      "Gender",
    "height":      "Height (cm)",
    "weight":      "Weight (kg)",
    "ap_hi":       "Systolic BP",
    "ap_lo":       "Diastolic BP",
    "cholesterol": "Cholesterol",
    "gluc":        "Glucose",
    "smoke":       "Smoker",
    "alco":        "Alcohol",
    "active":      "Physically Active",
}

FEATURE_ORDER = list(FEATURE_LABELS.keys())

RISK_COLORS = {
    "High":     colors.HexColor("#e74c3c"),
    "Moderate": colors.HexColor("#f39c12"),
    "Low":      colors.HexColor("#2ecc71"),
}


def get_risk_factors(model, scaler, patient_data: dict) -> list[dict]:
    """
    Compute per-feature contribution to the prediction using
    logistic regression coefficients × scaled feature values.

    Returns a list of dicts sorted by absolute contribution (descending):
        [{"feature": "Systolic BP", "contribution": 0.42, "value": 140}, ...]
    """
    coefs = model.coef_[0]                          # shape: (n_features,)
    raw   = np.array([patient_data[f] for f in FEATURE_ORDER])
    scaled = scaler.transform(raw.reshape(1, -1))[0]
    contributions = coefs * scaled                  # element-wise product

    factors = []
    for i, feat in enumerate(FEATURE_ORDER):
        factors.append({
            "feature":      FEATURE_LABELS[feat],
            "raw_value":    patient_data[feat],
            "contribution": float(contributions[i]),
        })

    # Sort by absolute contribution descending
    factors.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return factors


def generate_pdf_report(patient_data: dict, result: dict,
                        risk_factors: list[dict], username: str) -> bytes:
    """
    Build a PDF report and return it as bytes.

    Args:
        patient_data: Raw patient input dict.
        result: Prediction result from predict_risk().
        risk_factors: Output of get_risk_factors().
        username: Logged-in user's name.

    Returns:
        PDF file as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    risk_color = RISK_COLORS.get(result["risk_level"], colors.grey)

    title_style = ParagraphStyle(
        "title", parent=styles["Title"],
        fontSize=22, textColor=colors.HexColor("#c0392b"),
        spaceAfter=4, alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        "subtitle", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#718096"),
        alignment=TA_CENTER, spaceAfter=16
    )
    section_style = ParagraphStyle(
        "section", parent=styles["Heading2"],
        fontSize=12, textColor=colors.HexColor("#2d3748"),
        spaceBefore=14, spaceAfter=6,
        borderPad=4
    )
    normal = styles["Normal"]

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("❤ CardioRisk — Prediction Report", title_style))
    story.append(Paragraph(f"Generated for: {username}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#e2e8f0"), spaceAfter=12))

    # ── Risk result banner ────────────────────────────────────────────────────
    risk_table = Table(
        [[
            Paragraph(f"Risk Level: {result['risk_level']}", ParagraphStyle(
                "risk", fontSize=16, textColor=colors.white,
                alignment=TA_CENTER, fontName="Helvetica-Bold"
            )),
            Paragraph(f"Probability: {result['risk_probability']*100:.1f}%", ParagraphStyle(
                "prob", fontSize=16, textColor=colors.white,
                alignment=TA_CENTER, fontName="Helvetica-Bold"
            )),
        ]],
        colWidths=["50%", "50%"]
    )
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_color),
        ("ROUNDEDCORNERS", [8]),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 14))

    diagnosis = ("Cardiovascular disease risk detected."
                 if result["risk_label"] == 1
                 else "No significant cardiovascular risk detected.")
    story.append(Paragraph(f"<b>Diagnosis:</b> {diagnosis}", normal))
    story.append(Spacer(1, 10))

    # ── Patient data ──────────────────────────────────────────────────────────
    story.append(Paragraph("Patient Information", section_style))

    gender_map = {1: "Female", 2: "Male"}
    chol_map   = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
    yn_map     = {0: "No", 1: "Yes"}

    patient_rows = [
        ["Field", "Value"],
        ["Age",              f"{patient_data['age']} years"],
        ["Gender",           gender_map.get(patient_data["gender"], "-")],
        ["Height",           f"{patient_data['height']} cm"],
        ["Weight",           f"{patient_data['weight']} kg"],
        ["Systolic BP",      f"{patient_data['ap_hi']} mmHg"],
        ["Diastolic BP",     f"{patient_data['ap_lo']} mmHg"],
        ["Cholesterol",      chol_map.get(patient_data["cholesterol"], "-")],
        ["Glucose",          chol_map.get(patient_data["gluc"], "-")],
        ["Smoker",           yn_map.get(patient_data["smoke"], "-")],
        ["Alcohol",          yn_map.get(patient_data["alco"], "-")],
        ["Physically Active",yn_map.get(patient_data["active"], "-")],
    ]

    pt = Table(patient_rows, colWidths=["45%", "55%"])
    pt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#2d3748")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.HexColor("#f7fafc"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(pt)

    # ── Risk factor breakdown ─────────────────────────────────────────────────
    story.append(Paragraph("Risk Factor Breakdown", section_style))
    story.append(Paragraph(
        "Shows which factors contributed most to this prediction. "
        "Positive values increase risk; negative values reduce it.",
        ParagraphStyle("hint", parent=normal, fontSize=9,
                       textColor=colors.HexColor("#718096"), spaceAfter=8)
    ))

    rf_rows = [["Factor", "Patient Value", "Contribution"]]
    for f in risk_factors[:8]:   # top 8
        direction = "▲ Increases risk" if f["contribution"] > 0 else "▼ Reduces risk"
        rf_rows.append([
            f["feature"],
            str(f["raw_value"]),
            f"{f['contribution']:+.3f}  ({direction})"
        ])

    rft = Table(rf_rows, colWidths=["35%", "25%", "40%"])
    rft.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#c0392b")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.HexColor("#fff5f5"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(rft)

    # ── Recommendations ───────────────────────────────────────────────────────
    story.append(Paragraph("General Recommendations", section_style))
    recs = _get_recommendations(result["risk_level"], patient_data)
    for rec in recs:
        story.append(Paragraph(f"• {rec}", ParagraphStyle(
            "rec", parent=normal, fontSize=10, spaceAfter=4,
            leftIndent=10
        )))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph(
        "This report is generated by CardioRisk and is for informational purposes only. "
        "Consult a qualified medical professional for diagnosis and treatment.",
        ParagraphStyle("footer", parent=normal, fontSize=8,
                       textColor=colors.HexColor("#a0aec0"),
                       alignment=TA_CENTER, spaceBefore=8)
    ))

    doc.build(story)
    return buffer.getvalue()


def _get_recommendations(risk_level: str, patient_data: dict) -> list[str]:
    """Return tailored recommendations based on risk level and patient data."""
    recs = []
    if risk_level == "High":
        recs.append("Consult a cardiologist as soon as possible.")
        recs.append("Monitor blood pressure daily and keep a log.")
    elif risk_level == "Moderate":
        recs.append("Schedule a check-up with your doctor within the next month.")
        recs.append("Monitor blood pressure regularly.")
    else:
        recs.append("Maintain your current healthy lifestyle.")
        recs.append("Continue regular health check-ups annually.")

    if patient_data.get("smoke") == 1:
        recs.append("Quitting smoking significantly reduces cardiovascular risk.")
    if patient_data.get("alco") == 1:
        recs.append("Reducing alcohol consumption improves heart health.")
    if patient_data.get("active") == 0:
        recs.append("Aim for at least 30 minutes of moderate exercise 5 days a week.")
    if patient_data.get("cholesterol", 1) >= 2:
        recs.append("Follow a low-cholesterol diet and consult your doctor about cholesterol levels.")
    if patient_data.get("ap_hi", 0) >= 140:
        recs.append("Your systolic BP is elevated. Reduce salt intake and manage stress.")

    return recs
