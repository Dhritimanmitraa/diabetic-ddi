"""
Pydantic schemas for Diabetic Patient DDI Module.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class DiabetesTypeEnum(str, Enum):
    """Type of diabetes."""
    TYPE_1 = "type_1"
    TYPE_2 = "type_2"
    GESTATIONAL = "gestational"
    PREDIABETES = "prediabetes"
    OTHER = "other"


class RiskLevelEnum(str, Enum):
    """Risk level for drug."""
    SAFE = "safe"
    CAUTION = "caution"
    HIGH_RISK = "high_risk"
    CONTRAINDICATED = "contraindicated"
    FATAL = "fatal"


# Patient Schemas
class PatientLabsBase(BaseModel):
    """Lab values for a diabetic patient."""
    hba1c: Optional[float] = Field(None, description="HbA1c percentage (e.g., 7.5)")
    fasting_glucose: Optional[float] = Field(None, description="Fasting glucose mg/dL")
    egfr: Optional[float] = Field(None, description="eGFR mL/min/1.73m²")
    creatinine: Optional[float] = Field(None, description="Creatinine mg/dL")
    potassium: Optional[float] = Field(None, description="Potassium mEq/L")
    alt: Optional[float] = Field(None, description="ALT U/L")
    ast: Optional[float] = Field(None, description="AST U/L")


class PatientComplicationsBase(BaseModel):
    """Diabetes complications flags."""
    has_nephropathy: bool = False
    has_retinopathy: bool = False
    has_neuropathy: bool = False
    has_cardiovascular: bool = False
    has_hypertension: bool = False
    has_hyperlipidemia: bool = False
    has_obesity: bool = False


class DiabeticPatientCreate(BaseModel):
    """Create a new diabetic patient profile."""
    patient_id: str = Field(..., min_length=1, description="Unique patient identifier")
    name: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = None
    weight_kg: Optional[float] = Field(None, gt=0)
    height_cm: Optional[float] = Field(None, gt=0)
    
    diabetes_type: DiabetesTypeEnum = DiabetesTypeEnum.TYPE_2
    years_with_diabetes: Optional[int] = Field(None, ge=0)
    
    # Labs
    labs: Optional[PatientLabsBase] = None
    
    # Complications
    complications: Optional[PatientComplicationsBase] = None
    
    # Other
    allergies: Optional[List[str]] = None
    comorbidities: Optional[List[str]] = None


class DiabeticPatientUpdate(BaseModel):
    """Update patient profile."""
    name: Optional[str] = None
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    
    diabetes_type: Optional[DiabetesTypeEnum] = None
    years_with_diabetes: Optional[int] = None
    
    labs: Optional[PatientLabsBase] = None
    complications: Optional[PatientComplicationsBase] = None
    
    allergies: Optional[List[str]] = None
    comorbidities: Optional[List[str]] = None


class DiabeticPatientResponse(BaseModel):
    """Patient profile response."""
    id: int
    patient_id: str
    name: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    weight_kg: Optional[float]
    height_cm: Optional[float]
    bmi: Optional[float]
    
    diabetes_type: str
    years_with_diabetes: Optional[int]
    
    # Labs
    hba1c: Optional[float]
    fasting_glucose: Optional[float]
    egfr: Optional[float]
    kidney_stage: Optional[str]
    creatinine: Optional[float]
    potassium: Optional[float]
    alt: Optional[float]
    ast: Optional[float]
    
    # Complications
    has_nephropathy: bool
    has_retinopathy: bool
    has_neuropathy: bool
    has_cardiovascular: bool
    has_hypertension: bool
    has_hyperlipidemia: bool
    has_obesity: bool
    
    # Other
    allergies: Optional[List[str]]
    comorbidities: Optional[List[str]]
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Medication Schemas
class MedicationCreate(BaseModel):
    """Add medication to patient."""
    drug_name: str = Field(..., min_length=1)
    generic_name: Optional[str] = None
    drug_class: Optional[str] = None
    dose: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    indication: Optional[str] = None
    is_diabetes_medication: bool = False


class MedicationResponse(BaseModel):
    """Medication response."""
    id: int
    drug_name: str
    generic_name: Optional[str]
    drug_class: Optional[str]
    dose: Optional[str]
    frequency: Optional[str]
    route: Optional[str]
    indication: Optional[str]
    is_diabetes_medication: bool
    is_active: bool
    start_date: Optional[datetime]
    
    class Config:
        from_attributes = True


# Risk Assessment Schemas
class DrugRiskCheckRequest(BaseModel):
    """Request to check a drug's risk for a patient."""
    patient_id: str = Field(..., description="Patient identifier")
    drug_name: str = Field(..., min_length=1, description="Drug to check")


class DrugRiskCheckResponse(BaseModel):
    """Drug risk assessment result."""
    drug_name: str
    risk_level: RiskLevelEnum
    risk_score: float = Field(..., ge=0, le=100)
    severity: str
    risk_factors: List[str]
    rule_references: List[str] = []
    evidence_sources: List[str] = []
    patient_factors: List[str] = []
    recommendation: str
    alternatives: List[str]
    monitoring: List[str]
    interactions: List[Dict]
    
    # Visual indicators
    is_safe: bool
    is_fatal: bool
    requires_monitoring: bool

    # ML (diabetic model) metadata
    ml_risk_level: Optional[RiskLevelEnum] = None
    ml_probability: Optional[float] = None
    ml_decision_source: Optional[str] = None  # ml_primary, rule_override, rules_only
    ml_model_version: Optional[str] = None

    # Explainability (SHAP + LLM)
    shap_explanation: Optional[Dict] = Field(
        None, 
        description="SHAP-based feature attributions showing why ML made this prediction"
    )
    llm_explanation: Optional[str] = Field(
        None,
        description="Patient-friendly explanation of the risk assessment"
    )
    llm_analysis: Optional[Dict] = Field(
        None,
        description="LLM-based drug risk analysis (runs in parallel with ML)"
    )


class MedicationListCheckRequest(BaseModel):
    """Check all medications for a patient."""
    patient_id: str
    medications: Optional[List[str]] = None  # If None, use patient's current meds


class MedicationListCheckResponse(BaseModel):
    """Results for checking medication list."""
    patient_id: str
    total_medications: int
    safe_count: int
    caution_count: int
    high_risk_count: int
    contraindicated_count: int
    fatal_count: int
    
    assessments: List[DrugRiskCheckResponse]
    
    # Overall recommendation
    overall_risk_level: str
    critical_alerts: List[str]
    recommendations: List[str]


class SafeAlternativesRequest(BaseModel):
    """Request for safe alternatives."""
    patient_id: str
    drug_name: str


class SafeAlternativeResponse(BaseModel):
    """A safe alternative drug."""
    drug: str
    risk_level: str
    risk_score: float
    considerations: List[str]


class SafeAlternativesResponse(BaseModel):
    """Response with safe alternatives."""
    original_drug: str
    original_risk_level: str
    alternatives: List[SafeAlternativeResponse]


# Report Schemas
class PatientDDIReportRequest(BaseModel):
    """Request a full DDI report for a patient."""
    patient_id: str
    include_alternatives: bool = True
    include_monitoring_plan: bool = True


class PatientDDIReportResponse(BaseModel):
    """Full DDI report for a patient."""
    patient: DiabeticPatientResponse
    report_generated_at: datetime
    
    # Current medications
    current_medications: List[MedicationResponse]
    
    # Risk summary
    medication_assessments: List[DrugRiskCheckResponse]
    
    # Alerts
    fatal_risks: List[Dict]
    contraindicated_drugs: List[Dict]
    high_risk_drugs: List[Dict]
    
    # Recommendations
    recommended_alternatives: Dict[str, List[SafeAlternativeResponse]]
    monitoring_plan: List[str]
    
    # Summary
    overall_safety_score: float
    action_required: bool
    summary: str


# Preview / simulation schemas
class PreviewPatientContext(BaseModel):
    diabetes_type: Optional[str] = "type_2"
    age: Optional[int] = None
    hba1c: Optional[float] = None
    fasting_glucose: Optional[float] = None
    egfr: Optional[float] = None
    creatinine: Optional[float] = None
    potassium: Optional[float] = None
    alt: Optional[float] = None
    ast: Optional[float] = None
    has_nephropathy: bool = False
    has_retinopathy: bool = False
    has_neuropathy: bool = False
    has_cardiovascular: bool = False
    has_hypertension: bool = False
    has_hyperlipidemia: bool = False
    has_obesity: bool = False


class RulesPreviewRequest(BaseModel):
    patient: PreviewPatientContext
    drugs: List[str]


class RulesPreviewResponse(BaseModel):
    assessments: List[DrugRiskCheckResponse]

