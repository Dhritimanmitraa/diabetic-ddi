"""
Database models for Diabetic Patient DDI Module.

Stores patient profiles, medications, and risk assessments.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.database import Base


class DiabetesType(str, enum.Enum):
    """Type of diabetes."""
    TYPE_1 = "type_1"
    TYPE_2 = "type_2"
    GESTATIONAL = "gestational"
    PREDIABETES = "prediabetes"
    OTHER = "other"


class RiskLevel(str, enum.Enum):
    """Risk level for drug in diabetic patient."""
    SAFE = "safe"
    CAUTION = "caution"
    HIGH_RISK = "high_risk"
    CONTRAINDICATED = "contraindicated"
    FATAL = "fatal"


class DiabeticPatient(Base):
    """Patient profile for diabetic DDI analysis."""
    __tablename__ = "diabetic_patients"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Basic info
    patient_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    weight_kg = Column(Float, nullable=True)
    height_cm = Column(Float, nullable=True)
    
    # Diabetes-specific
    diabetes_type = Column(String(50), default=DiabetesType.TYPE_2.value)
    diagnosis_date = Column(DateTime, nullable=True)
    years_with_diabetes = Column(Integer, nullable=True)
    
    # Key labs
    hba1c = Column(Float, nullable=True)  # Glycated hemoglobin (%)
    fasting_glucose = Column(Float, nullable=True)  # mg/dL
    egfr = Column(Float, nullable=True)  # Kidney function (mL/min/1.73m²)
    creatinine = Column(Float, nullable=True)  # mg/dL
    potassium = Column(Float, nullable=True)  # mEq/L
    alt = Column(Float, nullable=True)  # Liver function
    ast = Column(Float, nullable=True)  # Liver function
    
    # Complications flags
    has_nephropathy = Column(Boolean, default=False)  # Kidney disease
    has_retinopathy = Column(Boolean, default=False)  # Eye disease
    has_neuropathy = Column(Boolean, default=False)  # Nerve damage
    has_cardiovascular = Column(Boolean, default=False)  # Heart disease
    has_hypertension = Column(Boolean, default=False)
    has_hyperlipidemia = Column(Boolean, default=False)
    has_obesity = Column(Boolean, default=False)
    
    # Other conditions
    allergies = Column(Text, nullable=True)  # JSON list
    comorbidities = Column(Text, nullable=True)  # JSON list
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    medications = relationship("DiabeticMedication", back_populates="patient", cascade="all, delete-orphan")
    risk_assessments = relationship("DiabeticDrugRisk", back_populates="patient", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<DiabeticPatient(id={self.patient_id}, type={self.diabetes_type}, HbA1c={self.hba1c})>"
    
    @property
    def bmi(self) -> float:
        """Calculate BMI."""
        if self.weight_kg and self.height_cm:
            height_m = self.height_cm / 100
            return round(self.weight_kg / (height_m ** 2), 1)
        return None
    
    @property
    def kidney_stage(self) -> str:
        """Estimate CKD stage from eGFR."""
        if not self.egfr:
            return "unknown"
        if self.egfr >= 90:
            return "normal"
        elif self.egfr >= 60:
            return "stage_2"
        elif self.egfr >= 45:
            return "stage_3a"
        elif self.egfr >= 30:
            return "stage_3b"
        elif self.egfr >= 15:
            return "stage_4"
        else:
            return "stage_5"


class DiabeticMedication(Base):
    """Current medications for a diabetic patient."""
    __tablename__ = "diabetic_medications"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("diabetic_patients.id"), nullable=False)
    
    # Drug info
    drug_name = Column(String(255), nullable=False)
    drug_id = Column(Integer, ForeignKey("drugs.id"), nullable=True)
    generic_name = Column(String(255), nullable=True)
    drug_class = Column(String(100), nullable=True)
    
    # Dosage
    dose = Column(String(100), nullable=True)
    frequency = Column(String(100), nullable=True)
    route = Column(String(50), nullable=True)  # oral, injection, etc.
    
    # Purpose
    indication = Column(String(255), nullable=True)  # Why prescribed
    is_diabetes_medication = Column(Boolean, default=False)
    
    # Status
    is_active = Column(Boolean, default=True)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    patient = relationship("DiabeticPatient", back_populates="medications")
    
    def __repr__(self):
        return f"<DiabeticMedication({self.drug_name}, patient={self.patient_id})>"


class DiabeticDrugRisk(Base):
    """Risk assessment for a drug in a specific diabetic patient."""
    __tablename__ = "diabetic_drug_risks"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("diabetic_patients.id"), nullable=False)
    
    # Drug info
    drug_name = Column(String(255), nullable=False)
    drug_id = Column(Integer, ForeignKey("drugs.id"), nullable=True)
    
    # Risk assessment
    risk_level = Column(String(50), nullable=False)  # safe, caution, high_risk, contraindicated, fatal
    risk_score = Column(Float, nullable=True)  # 0-100
    
    # Reasons
    risk_factors = Column(Text, nullable=True)  # JSON list of risk factors
    contraindication_reason = Column(Text, nullable=True)
    
    # Recommendations
    recommendation = Column(Text, nullable=True)
    alternative_drugs = Column(Text, nullable=True)  # JSON list
    monitoring_required = Column(Text, nullable=True)  # What to monitor
    
    # Interactions with current meds
    interacting_medications = Column(Text, nullable=True)  # JSON list
    interaction_severity = Column(String(50), nullable=True)
    
    # Context
    assessed_at = Column(DateTime, default=datetime.utcnow)
    labs_at_assessment = Column(Text, nullable=True)  # JSON snapshot of labs
    
    # Relationships
    patient = relationship("DiabeticPatient", back_populates="risk_assessments")
    
    def __repr__(self):
        return f"<DiabeticDrugRisk({self.drug_name}, risk={self.risk_level})>"


class DiabeticDrugRule(Base):
    """Curated rules for drugs in diabetic patients."""
    __tablename__ = "diabetic_drug_rules"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Drug identification
    drug_name = Column(String(255), nullable=False, index=True)
    drug_class = Column(String(100), nullable=True)
    
    # Base risk for diabetics
    base_risk_level = Column(String(50), nullable=False)
    
    # Condition-specific risks
    risk_with_nephropathy = Column(String(50), nullable=True)
    risk_with_cardiovascular = Column(String(50), nullable=True)
    risk_with_hypoglycemia_prone = Column(String(50), nullable=True)
    
    # Lab thresholds
    contraindicated_if_egfr_below = Column(Float, nullable=True)
    caution_if_egfr_below = Column(Float, nullable=True)
    contraindicated_if_potassium_above = Column(Float, nullable=True)
    
    # Effects on diabetes
    affects_blood_glucose = Column(Boolean, default=False)
    glucose_effect = Column(String(50), nullable=True)  # increases, decreases, variable
    masks_hypoglycemia = Column(Boolean, default=False)
    
    # Warnings and guidance
    warning = Column(Text, nullable=True)
    monitoring_guidance = Column(Text, nullable=True)
    dose_adjustment_guidance = Column(Text, nullable=True)
    
    # Alternatives
    safer_alternatives = Column(Text, nullable=True)  # JSON list
    
    # Source
    source = Column(String(255), nullable=True)
    evidence_level = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<DiabeticDrugRule({self.drug_name}, base_risk={self.base_risk_level})>"

