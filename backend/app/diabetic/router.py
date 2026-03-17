"""
Diabetic DDI API Router.

Provides endpoints for managing diabetic patient profiles, drug risk assessments,
and PDF report generation.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional, Dict
from datetime import datetime
import logging
import json
import difflib

from app.database import get_db
from app.diabetic.service import DiabeticDDIService, create_diabetic_service
from app.diabetic.ml_predictor import get_diabetic_predictor
from app.diabetic.schemas import (
    DiabeticPatientCreate, DiabeticPatientUpdate, DiabeticPatientResponse,
    MedicationCreate, MedicationResponse,
    DrugRiskCheckRequest, DrugRiskCheckResponse,
    MedicationListCheckRequest, MedicationListCheckResponse,
    SafeAlternativesRequest, SafeAlternativesResponse,
    PatientDDIReportRequest, PatientDDIReportResponse,
    RulesPreviewRequest, RulesPreviewResponse,
)
from app.schemas import DrugResponse
from app.models import Drug, OffsidesEffect, TwosidesInteraction, User
from app.services.jwt_auth import require_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/diabetic", tags=["Diabetic DDI"])


async def get_service(db: AsyncSession = Depends(get_db)) -> DiabeticDDIService:
    """Dependency to get diabetic service."""
    return create_diabetic_service(db)


# ==================== Patient Endpoints ====================

@router.post("/patients", response_model=DiabeticPatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    data: DiabeticPatientCreate,
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Create a new diabetic patient profile.
    
    Include diabetes type, labs (HbA1c, eGFR, etc.), and complications
    for accurate drug risk assessment.
    """
    try:
        patient = await service.create_patient(data, user_id=current_user.id)
        return service._patient_to_response(patient)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/patients", response_model=List[DiabeticPatientResponse])
async def list_patients(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """List all diabetic patients with pagination."""
    patients, total = await service.list_patients(limit, offset, user_id=current_user.id)
    return [service._patient_to_response(p) for p in patients]


@router.get("/patients/{patient_id}", response_model=DiabeticPatientResponse)
async def get_patient(
    patient_id: str,
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """Get a specific patient profile by ID."""
    patient = await service.get_patient(patient_id, user_id=current_user.id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return service._patient_to_response(patient)


@router.patch("/patients/{patient_id}", response_model=DiabeticPatientResponse)
async def update_patient(
    patient_id: str,
    data: DiabeticPatientUpdate,
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Update patient profile.
    
    Update labs, complications, or other patient data.
    This will affect future drug risk assessments.
    """
    patient = await service.update_patient(patient_id, data, user_id=current_user.id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return service._patient_to_response(patient)


@router.delete("/patients/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: str,
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """Delete a patient and all associated data."""
    deleted = await service.delete_patient(patient_id, user_id=current_user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")


# ==================== Medication Endpoints ====================

@router.post("/patients/{patient_id}/medications", response_model=MedicationResponse, status_code=status.HTTP_201_CREATED)
async def add_medication(
    patient_id: str,
    data: MedicationCreate,
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """Add a medication to a patient's profile."""
    medication = await service.add_medication(patient_id, data, user_id=current_user.id)
    if not medication:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return MedicationResponse.model_validate(medication)


@router.get("/patients/{patient_id}/medications", response_model=List[MedicationResponse])
async def get_medications(
    patient_id: str,
    active_only: bool = Query(True),
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """Get all medications for a patient."""
    medications = await service.get_patient_medications(
        patient_id,
        active_only,
        user_id=current_user.id,
    )
    return [MedicationResponse.model_validate(m) for m in medications]


@router.delete("/patients/{patient_id}/medications/{medication_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_medication(
    patient_id: str,
    medication_id: int,
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """Remove a medication from a patient's profile."""
    deleted = await service.remove_medication(patient_id, medication_id, user_id=current_user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Medication not found")


# ==================== Risk Assessment Endpoints ====================

@router.post("/risk-check", response_model=DrugRiskCheckResponse)
async def check_drug_risk(
    data: DrugRiskCheckRequest,
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Check the risk of a drug for a specific diabetic patient.
    
    Returns IMMEDIATELY with rules + ML results (fast).
    LLM analysis is fetched separately via /risk-check/llm endpoint.
    
    Takes into account:
    - Patient's diabetes type and complications
    - Lab values (eGFR, potassium, liver enzymes)
    - Current medications (for interactions)
    - Drug-specific risks in diabetics
    
    Returns risk level (safe/caution/high_risk/contraindicated/fatal),
    risk factors, recommendations, and safer alternatives.
    """
    result = await service.check_drug_risk(
        data.patient_id,
        data.drug_name,
        user_id=current_user.id,
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Patient {data.patient_id} not found")
    return result


@router.post("/risk-check/llm", response_model=Dict)
async def get_llm_analysis(
    data: DrugRiskCheckRequest,
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Get LLM analysis for a drug risk check (called separately after initial response).
    
    This endpoint is called in the background after the main risk-check endpoint
    returns, allowing the UI to show results immediately and update when LLM is ready.
    """
    patient = await service.get_patient(data.patient_id, user_id=current_user.id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {data.patient_id} not found")
    
    # Get current medications
    medications = await service.get_patient_medications(data.patient_id, user_id=current_user.id)
    current_meds = [m.drug_name for m in medications]
    
    # Build patient context
    patient_context = service._build_patient_context(patient)
    
    # Get LLM analysis
    try:
        llm_result = await service.llm_checker.check_drug_risk(
            data.drug_name, patient_context, current_meds
        )
        
        if llm_result:
            return {
                "llm_analysis": {
                    "risk_level": llm_result.risk_level,
                    "risk_score": llm_result.risk_score,
                    "reasoning": llm_result.reasoning,
                    "key_concerns": llm_result.key_concerns,
                    "monitoring_needed": llm_result.monitoring_needed,
                    "model_used": llm_result.model_used,
                    "was_fallback": llm_result.was_fallback,
                }
            }
        else:
            return {"llm_analysis": None, "error": "LLM analysis unavailable"}
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return {"llm_analysis": None, "error": str(e)}


@router.post("/medication-list-check", response_model=MedicationListCheckResponse)
async def check_medication_list(
    data: MedicationListCheckRequest,
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Check all medications in a patient's regimen.
    
    If medications list is not provided, uses the patient's current medications.
    Returns risk assessment for each drug plus overall recommendations.
    """
    result = await service.check_all_medications(
        data.patient_id,
        data.medications,
        user_id=current_user.id,
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Patient {data.patient_id} not found")
    return result


@router.post("/alternatives", response_model=SafeAlternativesResponse)
async def find_alternatives(
    data: SafeAlternativesRequest,
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Find safer alternatives for a drug.
    
    Suggests drugs from the same class that are safer for this specific
    diabetic patient based on their labs and complications.
    """
    result = await service.find_safe_alternatives(
        data.patient_id,
        data.drug_name,
        user_id=current_user.id,
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Patient {data.patient_id} not found")
    return result


# ==================== Report Endpoints ====================

@router.post("/report", response_model=PatientDDIReportResponse)
async def generate_report(
    data: PatientDDIReportRequest,
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Generate a comprehensive DDI report for a diabetic patient.
    
    Includes:
    - Patient profile summary
    - All current medications with risk assessments
    - Fatal and contraindicated drugs highlighted
    - Safer alternatives for risky drugs
    - Monitoring recommendations
    - Overall safety score
    """
    result = await service.generate_patient_report(
        data.patient_id,
        data.include_alternatives,
        user_id=current_user.id,
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Patient {data.patient_id} not found")
    return result


@router.get("/report/{patient_id}", response_model=PatientDDIReportResponse)
async def get_report(
    patient_id: str,
    include_alternatives: bool = Query(True),
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """Get a DDI report for a patient (GET variant)."""
    result = await service.generate_patient_report(
        patient_id,
        include_alternatives,
        user_id=current_user.id,
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return result


@router.get("/report/{patient_id}/pdf")
async def get_report_pdf(
    patient_id: str,
    include_alternatives: bool = Query(True),
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Generate and download a PDF report for a diabetic patient.
    
    Returns a downloadable PDF file containing:
    - Patient profile summary
    - Current medications
    - Risk assessments for each medication
    - Critical warnings for dangerous drugs
    - Overall safety score
    """
    from app.services.pdf_generator import generate_patient_report_pdf
    
    # Get the patient
    patient = await service.get_patient(patient_id, user_id=current_user.id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    # Generate report data
    report = await service.generate_patient_report(
        patient_id,
        include_alternatives,
        user_id=current_user.id,
    )
    if not report:
        raise HTTPException(status_code=404, detail=f"Could not generate report for patient {patient_id}")
    
    # Get medications
    medications = await service.get_patient_medications(patient_id, user_id=current_user.id)
    meds_data = [
        {
            "drug_name": m.drug_name,
            "dosage": m.dose or "N/A",
            "frequency": m.frequency or "N/A"
        }
        for m in medications
    ]
    
    # Build patient data dict
    patient_data = {
        "patient_id": patient.patient_id,
        "diabetes_type": patient.diabetes_type,
        "age": patient.age,
        "labs": {
            "egfr": patient.egfr,
            "hba1c": patient.hba1c,
            "potassium": patient.potassium,
            "creatinine": patient.creatinine,
        },
        "complications": []
    }
    if patient.has_nephropathy:
        patient_data["complications"].append("nephropathy")
    if patient.has_retinopathy:
        patient_data["complications"].append("retinopathy")
    if patient.has_neuropathy:
        patient_data["complications"].append("neuropathy")
    if patient.has_cardiovascular:
        patient_data["complications"].append("cardiovascular")
    
    # Build risk assessments
    risk_assessments = []
    if report.medication_assessments:
        for assessment in report.medication_assessments:
            risk_assessments.append({
                "drug_name": assessment.drug_name,
                "risk_level": assessment.risk_level,
                "risk_factors": assessment.risk_factors or [],
                "recommendations": assessment.recommendations or []
            })
    
    try:
        pdf_bytes = generate_patient_report_pdf(
            patient_data=patient_data,
            medications=meds_data,
            risk_assessments=risk_assessments,
            overall_score=report.overall_safety_score or 0
        )
        
        filename = f"DrugGuard_Report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="PDF generation requires reportlab. Install with: pip install reportlab"
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")


# ==================== Lab Report Analysis (Gemini Vision) ====================

@router.post("/analyze-report")
async def analyze_lab_report(
    file: UploadFile = File(..., description="Lab report image (JPEG, PNG) or PDF"),
    auto_create_patient: bool = Query(True, description="Automatically create patient from extracted data"),
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Analyze a lab report image using Google Gemini Vision AI.
    
    Extracts:
    - Patient demographics (name, age, gender)
    - Glucose values (HbA1c, FBS, PPBS, mean BG)
    - Kidney function (creatinine, eGFR)
    - Lipid profile (cholesterol, triglycerides, HDL, LDL, VLDL)
    - Liver function (ALT, AST)
    - Thyroid profile (T3, T4, TSH)
    
    Returns extracted values and AI-generated health summary.
    """
    from app.diabetic.lab_report_analyzer import get_lab_report_analyzer
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: JPEG, PNG, PDF. Got: {file.content_type}"
        )
    
    # Read file
    image_data = await file.read()
    
    if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large. Max size: 10MB")
    
    # Analyze with Gemini
    analyzer = get_lab_report_analyzer()
    result = await analyzer.analyze_report(image_data, file.filename or "report.jpg", file.content_type)
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Failed to analyze report")
    
    response_data = {
        "success": True,
        "extracted_values": result.extracted_values.model_dump(),
        "health_summary": result.health_summary.model_dump(),
        "model_used": result.model_used,
        "patient_created": False,
        "patient_id": None
    }
    
    # Auto-create patient if requested
    if auto_create_patient and result.extracted_values.patient.name:
        try:
            from app.diabetic.schemas import DiabeticPatientCreate, PatientLabsBase, PatientComplicationsBase
            import uuid
            
            # Generate patient ID if not found
            patient_id = result.extracted_values.patient.patient_id or f"RPT-{uuid.uuid4().hex[:8].upper()}"
            
            # Build patient data from extracted values
            labs = PatientLabsBase(
                hba1c=result.extracted_values.glucose.hba1c,
                fasting_glucose=result.extracted_values.glucose.fasting_glucose,
                postprandial_glucose=result.extracted_values.glucose.postprandial_glucose,
                mean_blood_glucose=result.extracted_values.glucose.mean_blood_glucose,
                egfr=result.extracted_values.kidney.egfr,
                creatinine=result.extracted_values.kidney.creatinine,
                potassium=result.extracted_values.potassium,
                total_cholesterol=result.extracted_values.lipid.total_cholesterol,
                triglycerides=result.extracted_values.lipid.triglycerides,
                hdl_cholesterol=result.extracted_values.lipid.hdl_cholesterol,
                ldl_cholesterol=result.extracted_values.lipid.ldl_cholesterol,
                vldl_cholesterol=result.extracted_values.lipid.vldl_cholesterol,
                alt=result.extracted_values.liver.alt,
                ast=result.extracted_values.liver.ast,
            )
            
            # Determine diabetes type from values
            diabetes_type = "type_2"  # Default
            if result.extracted_values.glucose.hba1c:
                if result.extracted_values.glucose.hba1c < 5.7:
                    diabetes_type = "other"  # Normal
                elif result.extracted_values.glucose.hba1c < 6.5:
                    diabetes_type = "prediabetes"
                else:
                    diabetes_type = "type_2"
            
            patient_data = DiabeticPatientCreate(
                patient_id=patient_id,
                name=result.extracted_values.patient.name,
                age=result.extracted_values.patient.age,
                gender=result.extracted_values.patient.gender,
                diabetes_type=diabetes_type,
                labs=labs,
                complications=PatientComplicationsBase()
            )
            
            # Try to delete existing patient with same ID
            await service.delete_patient(patient_id, user_id=current_user.id)
            
            # Create new patient
            patient = await service.create_patient(patient_data, user_id=current_user.id)
            response_data["patient_created"] = True
            response_data["patient_id"] = patient.patient_id
            
        except Exception as e:
            logger.warning(f"Failed to auto-create patient: {e}")
            response_data["patient_create_error"] = str(e)
    
    return response_data


@router.post("/analyze-report/personalized-ddi")
async def get_personalized_ddi_from_report(
    patient_id: str = Query(..., description="Patient ID to analyze"),
    drug_name: str = Query(..., description="Drug to check"),
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Get PERSONALIZED drug risk analysis based on patient's actual lab values.
    
    This uses AI to explain why THIS patient's specific lab values
    affect the safety of the drug. Results are unique to each patient.
    """
    from app.diabetic.lab_report_analyzer import get_lab_report_analyzer
    
    # Get patient
    patient = await service.get_patient(patient_id, user_id=current_user.id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    # Build patient data dict
    patient_data = {
        "name": patient.name or patient.patient_id,
        "age": patient.age,
        "gender": patient.gender,
        "diabetes_type": patient.diabetes_type,
        "hba1c": patient.hba1c,
        "fasting_glucose": patient.fasting_glucose,
        "egfr": patient.egfr,
        "creatinine": patient.creatinine,
        "potassium": patient.potassium,
        "total_cholesterol": getattr(patient, 'total_cholesterol', None),
        "triglycerides": getattr(patient, 'triglycerides', None),
    }
    
    # Get personalized analysis
    analyzer = get_lab_report_analyzer()
    result = await analyzer.get_personalized_ddi_analysis(patient_data, drug_name)
    
    return {
        "patient_id": patient_id,
        "patient_name": patient.name,
        "drug_name": drug_name,
        "personalized_analysis": result
    }


@router.get("/analyzer-status")
async def get_analyzer_status():
    """Check if the Gemini Vision analyzer is available."""
    from app.diabetic.lab_report_analyzer import get_lab_report_analyzer
    
    analyzer = get_lab_report_analyzer()
    return {
        "gemini_available": analyzer.gemini_available,
        "model": "gemini-1.5-flash",
        "features": [
            "lab_value_extraction",
            "health_summary_generation",
            "personalized_ddi_analysis"
        ]
    }


# ==================== Quick Check Endpoints ====================


@router.get("/quick-check/{patient_id}/{drug_name}", response_model=DrugRiskCheckResponse)
async def quick_drug_check(
    patient_id: str,
    drug_name: str,
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Quick drug risk check (GET endpoint).
    
    Convenient endpoint for checking a single drug without POST body.
    """
    result = await service.check_drug_risk(patient_id, drug_name, user_id=current_user.id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return result


# ==================== Info Endpoints ====================

@router.get("/rules/info")
async def get_rules_info():
    """Get information about the rules engine."""
    from app.diabetic.rules import DiabeticDrugRules
    rules = DiabeticDrugRules()
    
    return {
        "version": "1.0.0",
        "hypoglycemia_risk_drugs": sum(len(v) for v in rules.HYPOGLYCEMIA_RISK_DRUGS.values()),
        "hyperglycemia_risk_drugs": sum(len(v) for v in rules.HYPERGLYCEMIA_RISK_DRUGS.values()),
        "nephrotoxic_drugs": len(rules.NEPHROTOXIC_DRUGS),
        "egfr_contraindications": list(rules.EGFR_CONTRAINDICATIONS.keys()),
        "hyperkalemia_risk_drugs": len(rules.HYPERKALEMIA_RISK_DRUGS),
        "cardioprotective_drugs": len(rules.CARDIOPROTECTIVE_IN_DIABETES),
        "masks_hypoglycemia": rules.MASK_HYPOGLYCEMIA,
        "description": "Diabetic-specific drug safety rules based on ADA/AACE guidelines"
    }


@router.get("/model-info")
async def get_diabetic_model_info():
    """Return status of the diabetic ML model (if trained and loaded)."""
    predictor = get_diabetic_predictor()
    return {
        "loaded": predictor.is_loaded if predictor else False,
        "model_version": predictor.model_version if predictor else None,
        "model_path": predictor.model_path if predictor else None,
    }


@router.get("/drug-classes")
async def get_drug_classes():
    """Get categorized drug lists for diabetics."""
    from app.diabetic.rules import DiabeticDrugRules
    rules = DiabeticDrugRules()
    
    return {
        "hypoglycemia_risk": rules.HYPOGLYCEMIA_RISK_DRUGS,
        "hyperglycemia_risk": rules.HYPERGLYCEMIA_RISK_DRUGS,
        "nephrotoxic": rules.NEPHROTOXIC_DRUGS,
        "hyperkalemia_risk": rules.HYPERKALEMIA_RISK_DRUGS,
        "hepatotoxic": rules.HEPATOTOXIC_DRUGS,
        "cardioprotective": rules.CARDIOPROTECTIVE_IN_DIABETES,
        "weight_gain": rules.WEIGHT_GAIN_DRUGS,
        "masks_hypoglycemia": rules.MASK_HYPOGLYCEMIA,
    }


# ==================== TWOSIDES Stats ====================

@router.get("/twosides/count")
async def twosides_count(db: AsyncSession = Depends(get_db)):
    """Return total rows ingested from TWOSIDES/OffSIDES."""
    result = await db.execute(select(func.count(TwosidesInteraction.id)))
    count = result.scalar() or 0
    return {"twosides_rows": count}


@router.get("/offsides/count")
async def offsides_count(db: AsyncSession = Depends(get_db)):
    """Return total rows ingested from OffSIDES (single-drug effects)."""
    result = await db.execute(select(func.count(OffsidesEffect.id)))
    count = result.scalar() or 0
    return {"offsides_rows": count}


# ==================== Drug Search (diabetic-focused) ====================

@router.get("/drugs/search", response_model=List[DrugResponse])
async def search_diabetic_drugs(
    query: str,
    limit: int = Query(10, ge=1, le=50),
    exclude_topical: bool = Query(True, description="Exclude obvious topicals/ophthalmic"),
    db: AsyncSession = Depends(get_db)
):
    """
    Search drugs for diabetic workflow, backed by the local DB (real data fetched from APIs).
    """
    query_l = query.lower().strip()
    q = select(Drug).limit(200)  # small pool for fuzzy post-filter
    result = await db.execute(q)
    pool = result.scalars().all()

    candidates = []
    for d in pool:
        names = [d.name or ""]
        if d.generic_name:
            names.append(d.generic_name)
        if d.brand_names:
            try:
                brands = json.loads(d.brand_names)
                if isinstance(brands, list):
                    names.extend(brands)
            except (json.JSONDecodeError, TypeError, ValueError):
                # brand_names might be malformed or not valid JSON
                pass
        names_l = [n.lower() for n in names if n]

        # Exact/partial match first
        if any(query_l in n for n in names_l):
            candidates.append(d)
            continue

        # Fuzzy match fallback
        best = difflib.get_close_matches(query_l, names_l, n=1, cutoff=0.82)
        if best:
            candidates.append(d)

    # Dedupe and filter topicals
    seen = set()
    filtered = []
    topical_keywords = ["cream", "ointment", "ophthalmic", "nasal", "topical"]
    for d in candidates:
        if d.id in seen:
            continue
        seen.add(d.id)
        if exclude_topical and any(k in (d.name or "").lower() for k in topical_keywords):
            continue
        filtered.append(d)

    # Limit
    filtered = filtered[:limit]

    return [DrugResponse.model_validate(d) for d in filtered]


# ==================== Rules Preview / Simulation ====================

@router.post("/rules/preview", response_model=RulesPreviewResponse)
async def preview_rules(
    data: RulesPreviewRequest,
    service: DiabeticDDIService = Depends(get_service)
):
    """
    Simulate rule hits for an ad-hoc patient context and a list of drugs.
    Does not persist anything.
    """
    patient_ctx = {
        "diabetes_type": data.patient.diabetes_type,
        "years_with_diabetes": None,
        "age": data.patient.age,
        "hba1c": data.patient.hba1c,
        "fasting_glucose": data.patient.fasting_glucose,
        "egfr": data.patient.egfr,
        "creatinine": data.patient.creatinine,
        "potassium": data.patient.potassium,
        "alt": data.patient.alt,
        "ast": data.patient.ast,
        "has_nephropathy": data.patient.has_nephropathy,
        "has_retinopathy": data.patient.has_retinopathy,
        "has_neuropathy": data.patient.has_neuropathy,
        "has_cardiovascular": data.patient.has_cardiovascular,
        "has_hypertension": data.patient.has_hypertension,
        "has_hyperlipidemia": data.patient.has_hyperlipidemia,
        "has_obesity": data.patient.has_obesity,
        "bmi": None,
    }

    assessments = []
    for drug in data.drugs:
        other_meds = [m for m in data.drugs if m != drug]
        ra = service.rules.assess_drug_risk(drug, patient_ctx, other_meds)
        assessments.append(service._assessment_to_response(ra))

    return RulesPreviewResponse(assessments=assessments)


# ==================== Food Interaction Endpoints ====================

@router.get("/food-interactions/{drug_name}")
async def get_food_interactions(
    drug_name: str,
    min_severity: Optional[str] = Query(
        None,
        description="Minimum severity to include (contraindicated, major, moderate, minor)"
    )
):
    """
    Get food interactions for a specific drug.
    
    Returns all foods that may interact with this medication, including:
    - Severity level (contraindicated, major, moderate, minor)
    - Effect description
    - Recommendation for patient
    - Timing (ongoing, around_dose, with_dose)
    """
    from app.services.food_interactions import get_food_interaction_service
    
    service = get_food_interaction_service()
    result = service.get_interactions(drug_name, min_severity)
    
    return result.to_dict()


@router.get("/food-categories")
async def get_food_categories():
    """
    Get all food categories and their descriptions.
    
    Useful for educating patients about different types of food-drug interactions.
    """
    from app.services.food_interactions import get_food_interaction_service
    
    service = get_food_interaction_service()
    return {
        "categories": service.get_all_food_categories(),
        "total": len(service.food_categories)
    }


@router.get("/food-categories/{category_id}/drugs")
async def get_drugs_by_food_category(category_id: str):
    """
    Get all drugs that interact with a specific food category.
    
    For example, get all drugs that interact with grapefruit.
    """
    from app.services.food_interactions import get_food_interaction_service
    
    service = get_food_interaction_service()
    drugs = service.get_drugs_by_food_category(category_id)
    
    if not drugs:
        raise HTTPException(
            status_code=404,
            detail=f"Food category '{category_id}' not found or has no interactions"
        )
    
    category_info = service.food_categories.get(category_id)
    
    return {
        "category": {
            "id": category_id,
            "name": category_info.name if category_info else category_id,
            "description": category_info.description if category_info else "",
            "examples": category_info.examples if category_info else []
        },
        "drugs": drugs,
        "total": len(drugs)
    }


@router.get("/patients/{patient_id}/food-interactions")
async def get_patient_food_interactions(
    patient_id: str,
    service: DiabeticDDIService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Get all food interactions for a patient's current medications.
    
    Provides a comprehensive dietary guidance report for the patient.
    """
    from app.services.food_interactions import get_food_interaction_service
    
    # Get patient
    patient = await service.get_patient(patient_id, user_id=current_user.id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    # Get patient's current medications
    medications = await service.get_patient_medications(patient_id, user_id=current_user.id)
    med_names = [m.drug_name for m in medications]
    
    if not med_names:
        return {
            "patient_id": patient_id,
            "patient_name": patient.name,
            "medications_checked": [],
            "total_interactions_found": 0,
            "has_critical_interactions": False,
            "foods_to_avoid": [],
            "all_interactions": [],
            "summary": "No medications found for this patient."
        }
    
    # Check all medications
    food_service = get_food_interaction_service()
    result = food_service.check_patient_medications(med_names)
    
    return {
        "patient_id": patient_id,
        "patient_name": patient.name,
        **result
    }
