"""
PDF Report Generation Service for DrugGuard.

Generates professional PDF reports for diabetic patient DDI assessments.
"""
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def generate_patient_report_pdf(
    patient_data: Dict[str, Any],
    medications: List[Dict[str, Any]],
    risk_assessments: List[Dict[str, Any]],
    overall_score: float
) -> bytes:
    """
    Generate a PDF report for a diabetic patient's DDI assessment.
    
    Args:
        patient_data: Patient profile information
        medications: List of current medications
        risk_assessments: List of risk assessments for each medication
        overall_score: Overall safety score (0-100)
        
    Returns:
        PDF content as bytes
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        logger.error("reportlab not installed. Install with: pip install reportlab")
        raise ImportError("PDF generation requires reportlab. Install with: pip install reportlab")

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='DrugGuardTitle',
        fontSize=24,
        spaceAfter=20,
        textColor=colors.HexColor('#14b89a'),
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='SectionTitle',
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor('#1e293b'),
        fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='SubTitle',
        fontSize=11,
        spaceBefore=8,
        spaceAfter=5,
        textColor=colors.HexColor('#334155'),
        fontName='Helvetica-Bold'
    ))
    # Modify existing BodyText style instead of adding duplicate
    styles['BodyText'].fontSize = 10
    styles['BodyText'].spaceAfter = 5
    styles['BodyText'].textColor = colors.HexColor('#475569')
    styles['BodyText'].leading = 14
    
    styles.add(ParagraphStyle(
        name='WarningText',
        fontSize=10,
        spaceAfter=5,
        textColor=colors.HexColor('#dc2626'),
        fontName='Helvetica-Bold'
    ))

    elements = []

    # Header
    elements.append(Paragraph("DrugGuard", styles['DrugGuardTitle']))
    elements.append(Paragraph("Diabetic Drug Interaction Report", styles['Heading2']))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#14b89a')))
    elements.append(Spacer(1, 20))

    # Report metadata
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['BodyText']))
    elements.append(Spacer(1, 15))

    # Patient Profile Section
    elements.append(Paragraph("Patient Profile", styles['SectionTitle']))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.gray))
    
    patient_info = [
        ["Patient ID:", patient_data.get('patient_id', 'N/A')],
        ["Diabetes Type:", patient_data.get('diabetes_type', 'N/A').replace('_', ' ').title()],
        ["Age:", f"{patient_data.get('age', 'N/A')} years"],
    ]
    
    # Lab values
    labs = patient_data.get('labs', {})
    if labs:
        if labs.get('egfr'):
            patient_info.append(["eGFR:", f"{labs['egfr']} mL/min/1.73m²"])
        if labs.get('hba1c'):
            patient_info.append(["HbA1c:", f"{labs['hba1c']}%"])
        if labs.get('potassium'):
            patient_info.append(["Potassium:", f"{labs['potassium']} mEq/L"])
    
    patient_table = Table(patient_info, colWidths=[1.8*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#475569')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 15))

    # Complications
    complications = patient_data.get('complications', [])
    if complications:
        elements.append(Paragraph("Diabetes Complications:", styles['SubTitle']))
        comp_text = ", ".join([c.replace('_', ' ').title() for c in complications])
        elements.append(Paragraph(comp_text, styles['BodyText']))
        elements.append(Spacer(1, 15))

    # Overall Safety Score
    elements.append(Paragraph("Overall Safety Score", styles['SectionTitle']))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.gray))
    
    score_color = colors.HexColor('#22c55e') if overall_score >= 70 else (
        colors.HexColor('#eab308') if overall_score >= 40 else colors.HexColor('#ef4444')
    )
    score_text = f"<font color='{score_color}'><b>{overall_score:.0f}/100</b></font>"
    score_label = "Good" if overall_score >= 70 else ("Moderate Risk" if overall_score >= 40 else "High Risk")
    elements.append(Paragraph(f"Safety Score: {score_text} ({score_label})", styles['BodyText']))
    elements.append(Spacer(1, 20))

    # Current Medications Section
    elements.append(Paragraph("Current Medications", styles['SectionTitle']))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.gray))
    
    if medications:
        med_data = [["Drug Name", "Dosage", "Frequency"]]
        for med in medications:
            med_data.append([
                med.get('drug_name', 'N/A'),
                med.get('dosage', 'N/A'),
                med.get('frequency', 'N/A')
            ])
        
        med_table = Table(med_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        med_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#14b89a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ]))
        elements.append(med_table)
    else:
        elements.append(Paragraph("No medications currently recorded.", styles['BodyText']))
    elements.append(Spacer(1, 20))

    # Risk Assessments Section
    elements.append(Paragraph("Risk Assessments", styles['SectionTitle']))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.gray))
    
    if risk_assessments:
        for idx, assessment in enumerate(risk_assessments):
            drug_name = assessment.get('drug_name', 'Unknown Drug')
            risk_level = assessment.get('risk_level', 'unknown')
            
            # Color coding for risk levels
            risk_colors = {
                'safe': colors.HexColor('#22c55e'),
                'caution': colors.HexColor('#eab308'),
                'high_risk': colors.HexColor('#f97316'),
                'contraindicated': colors.HexColor('#ef4444'),
                'fatal': colors.HexColor('#7f1d1d'),
            }
            risk_color = risk_colors.get(risk_level, colors.gray)
            
            elements.append(Paragraph(
                f"<b>{drug_name}</b> - <font color='{risk_color}'>{risk_level.replace('_', ' ').upper()}</font>",
                styles['SubTitle']
            ))
            
            # Risk factors
            risk_factors = assessment.get('risk_factors', [])
            if risk_factors:
                for factor in risk_factors[:3]:  # Limit to 3 factors per drug
                    elements.append(Paragraph(f"• {factor}", styles['BodyText']))
            
            # Recommendations
            recommendations = assessment.get('recommendations', [])
            if recommendations:
                elements.append(Paragraph("<b>Recommendations:</b>", styles['BodyText']))
                for rec in recommendations[:2]:  # Limit to 2 recommendations
                    elements.append(Paragraph(f"  → {rec}", styles['BodyText']))
            
            elements.append(Spacer(1, 10))
    else:
        elements.append(Paragraph("No risk assessments available.", styles['BodyText']))

    # Dangerous Drugs Warning
    dangerous = [a for a in risk_assessments if a.get('risk_level') in ['contraindicated', 'fatal']]
    if dangerous:
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("CRITICAL WARNINGS", styles['SectionTitle']))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.red))
        for d in dangerous:
            elements.append(Paragraph(
                f"[!] {d.get('drug_name', 'Unknown')} is CONTRAINDICATED for this patient!",
                styles['WarningText']
            ))
        elements.append(Spacer(1, 10))

    # Footer / Disclaimer
    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "<i>This report is generated by DrugGuard and is intended for informational purposes only. "
        "Always consult with a healthcare professional before making medication decisions. "
        "DrugGuard is not a substitute for professional medical advice.</i>",
        ParagraphStyle(
            name='Disclaimer',
            fontSize=8,
            textColor=colors.HexColor('#94a3b8'),
            alignment=TA_CENTER
        )
    ))

    # Build PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


def generate_simple_report_pdf(title: str, content: Dict[str, Any]) -> bytes:
    """
    Generate a simple PDF report with basic content.
    Fallback if reportlab is not available or for simpler reports.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        raise ImportError("PDF generation requires reportlab. Install with: pip install reportlab")

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, title)

    # Content
    y_position = height - 100
    c.setFont("Helvetica", 10)
    
    for key, value in content.items():
        if y_position < 50:
            c.showPage()
            y_position = height - 50
        c.drawString(50, y_position, f"{key}: {value}")
        y_position -= 20

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes
