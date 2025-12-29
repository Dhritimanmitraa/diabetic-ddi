"""
Diabetic DDI API Router.

Provides endpoints for managing diabetic patient profiles and drug risk assessments.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
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
from app.models import Drug, TwosidesInteraction, OffsidesEffect

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/diabetic", tags=["Diabetic DDI"])


async def get_service(db: AsyncSession = Depends(get_db)) -> DiabeticDDIService:
    """Dependency to get diabetic service."""
    return create_diabetic_service(db)


# ==================== Patient Endpoints ====================

@router.post("/patients", response_model=DiabeticPatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    data: DiabeticPatientCreate,
    service: DiabeticDDIService = Depends(get_service)
):
    """
    Create a new diabetic patient profile.
    
    Include diabetes type, labs (HbA1c, eGFR, etc.), and complications
    for accurate drug risk assessment.
    """
    try:
        patient = await service.create_patient(data)
        return service._patient_to_response(patient)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/patients", response_model=List[DiabeticPatientResponse])
async def list_patients(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: DiabeticDDIService = Depends(get_service)
):
    """List all diabetic patients with pagination."""
    patients, total = await service.list_patients(limit, offset)
    return [service._patient_to_response(p) for p in patients]


@router.get("/patients/{patient_id}", response_model=DiabeticPatientResponse)
async def get_patient(
    patient_id: str,
    service: DiabeticDDIService = Depends(get_service)
):
    """Get a specific patient profile by ID."""
    patient = await service.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return service._patient_to_response(patient)


@router.patch("/patients/{patient_id}", response_model=DiabeticPatientResponse)
async def update_patient(
    patient_id: str,
    data: DiabeticPatientUpdate,
    service: DiabeticDDIService = Depends(get_service)
):
    """
    Update patient profile.
    
    Update labs, complications, or other patient data.
    This will affect future drug risk assessments.
    """
    patient = await service.update_patient(patient_id, data)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return service._patient_to_response(patient)


@router.delete("/patients/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: str,
    service: DiabeticDDIService = Depends(get_service)
):
    """Delete a patient and all associated data."""
    deleted = await service.delete_patient(patient_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")


# ==================== Medication Endpoints ====================

@router.post("/patients/{patient_id}/medications", response_model=MedicationResponse, status_code=status.HTTP_201_CREATED)
async def add_medication(
    patient_id: str,
    data: MedicationCreate,
    service: DiabeticDDIService = Depends(get_service)
):
    """Add a medication to a patient's profile."""
    medication = await service.add_medication(patient_id, data)
    if not medication:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return MedicationResponse.model_validate(medication)


@router.get("/patients/{patient_id}/medications", response_model=List[MedicationResponse])
async def get_medications(
    patient_id: str,
    active_only: bool = Query(True),
    service: DiabeticDDIService = Depends(get_service)
):
    """Get all medications for a patient."""
    medications = await service.get_patient_medications(patient_id, active_only)
    return [MedicationResponse.model_validate(m) for m in medications]


@router.delete("/patients/{patient_id}/medications/{medication_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_medication(
    patient_id: str,
    medication_id: int,
    service: DiabeticDDIService = Depends(get_service)
):
    """Remove a medication from a patient's profile."""
    deleted = await service.remove_medication(patient_id, medication_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Medication not found")


# ==================== Risk Assessment Endpoints ====================

@router.post("/risk-check", response_model=DrugRiskCheckResponse)
async def check_drug_risk(
    data: DrugRiskCheckRequest,
    service: DiabeticDDIService = Depends(get_service)
):
    """
    Check the risk of a drug for a specific diabetic patient.
    
    Takes into account:
    - Patient's diabetes type and complications
    - Lab values (eGFR, potassium, liver enzymes)
    - Current medications (for interactions)
    - Drug-specific risks in diabetics
    
    Returns risk level (safe/caution/high_risk/contraindicated/fatal),
    risk factors, recommendations, and safer alternatives.
    """
    result = await service.check_drug_risk(data.patient_id, data.drug_name)
    if not result:
        raise HTTPException(status_code=404, detail=f"Patient {data.patient_id} not found")
    return result


@router.post("/medication-list-check", response_model=MedicationListCheckResponse)
async def check_medication_list(
    data: MedicationListCheckRequest,
    service: DiabeticDDIService = Depends(get_service)
):
    """
    Check all medications in a patient's regimen.
    
    If medications list is not provided, uses the patient's current medications.
    Returns risk assessment for each drug plus overall recommendations.
    """
    result = await service.check_all_medications(data.patient_id, data.medications or [])
    if not result:
        raise HTTPException(status_code=404, detail=f"Patient {data.patient_id} not found")
    return result


@router.post("/alternatives", response_model=SafeAlternativesResponse)
async def find_alternatives(
    data: SafeAlternativesRequest,
    service: DiabeticDDIService = Depends(get_service)
):
    """
    Find safer alternatives for a drug.
    
    Suggests drugs from the same class that are safer for this specific
    diabetic patient based on their labs and complications.
    """
    result = await service.find_safe_alternatives(data.patient_id, data.drug_name)
    if not result:
        raise HTTPException(status_code=404, detail=f"Patient {data.patient_id} not found")
    return result


# ==================== Report Endpoints ====================

@router.post("/report", response_model=PatientDDIReportResponse)
async def generate_report(
    data: PatientDDIReportRequest,
    service: DiabeticDDIService = Depends(get_service)
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
        data.patient_id, data.include_alternatives
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Patient {data.patient_id} not found")
    return result


@router.get("/report/{patient_id}", response_model=PatientDDIReportResponse)
async def get_report(
    patient_id: str,
    include_alternatives: bool = Query(True),
    service: DiabeticDDIService = Depends(get_service)
):
    """Get a DDI report for a patient (GET variant)."""
    result = await service.generate_patient_report(patient_id, include_alternatives)
    if not result:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return result


# ==================== Quick Check Endpoints ====================

@router.get("/quick-check/{patient_id}/{drug_name}", response_model=DrugRiskCheckResponse)
async def quick_drug_check(
    patient_id: str,
    drug_name: str,
    service: DiabeticDDIService = Depends(get_service)
):
    """
    Quick drug risk check (GET endpoint).
    
    Convenient endpoint for checking a single drug without POST body.
    """
    result = await service.check_drug_risk(patient_id, drug_name)
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
        # Extract values from SQLAlchemy columns (type checker doesn't understand runtime behavior)
        # At runtime, these are actual string values, not Column objects
        drug_name_raw = getattr(d, 'name', None)
        drug_name: str = str(drug_name_raw) if drug_name_raw is not None else ""
        names = [drug_name]
        generic_name_raw = getattr(d, 'generic_name', None)
        if generic_name_raw is not None:
            generic_name_val: str = str(generic_name_raw)
            names.append(generic_name_val)
        brand_names_raw = getattr(d, 'brand_names', None)
        if brand_names_raw is not None:
            brand_names_val: str = str(brand_names_raw)
            try:
                brands = json.loads(brand_names_val)
                if isinstance(brands, list):
                    names.extend(brands)
            except Exception:
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

