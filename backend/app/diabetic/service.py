"""
Diabetic DDI Service.

Main service for managing diabetic patient profiles and drug risk assessments.
"""
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update, delete
from sqlalchemy.orm import selectinload
import logging

from app.diabetic.models import (
    DiabeticPatient, DiabeticMedication, DiabeticDrugRisk, DiabeticDrugRule
)
from app.diabetic.rules import DiabeticDrugRules, RiskAssessment
from app.diabetic.schemas import (
    DiabeticPatientCreate, DiabeticPatientUpdate, DiabeticPatientResponse,
    MedicationCreate, MedicationResponse,
    DrugRiskCheckResponse, MedicationListCheckResponse,
    SafeAlternativeResponse, SafeAlternativesResponse,
    PatientDDIReportResponse, RiskLevelEnum
)
from app.diabetic.ml_predictor import get_diabetic_predictor

logger = logging.getLogger(__name__)


class DiabeticDDIService:
    """
    Service for diabetic patient drug interaction analysis.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.rules = DiabeticDrugRules()
        self.ml_predictor = get_diabetic_predictor()
    
    # ==================== Patient Management ====================
    
    async def create_patient(self, data: DiabeticPatientCreate) -> DiabeticPatient:
        """Create a new diabetic patient profile."""
        # Check if patient_id already exists
        existing = await self.db.execute(
            select(DiabeticPatient).where(DiabeticPatient.patient_id == data.patient_id)
        )
        if existing.scalar_one_or_none():
            raise ValueError(f"Patient {data.patient_id} already exists")
        
        patient = DiabeticPatient(
            patient_id=data.patient_id,
            name=data.name,
            age=data.age,
            gender=data.gender,
            weight_kg=data.weight_kg,
            height_cm=data.height_cm,
            diabetes_type=data.diabetes_type.value,
            years_with_diabetes=data.years_with_diabetes,
            allergies=json.dumps(data.allergies) if data.allergies else None,
            comorbidities=json.dumps(data.comorbidities) if data.comorbidities else None,
        )
        
        # Set labs if provided
        if data.labs:
            patient.hba1c = data.labs.hba1c
            patient.fasting_glucose = data.labs.fasting_glucose
            patient.egfr = data.labs.egfr
            patient.creatinine = data.labs.creatinine
            patient.potassium = data.labs.potassium
            patient.alt = data.labs.alt
            patient.ast = data.labs.ast
        
        # Set complications if provided
        if data.complications:
            patient.has_nephropathy = data.complications.has_nephropathy
            patient.has_retinopathy = data.complications.has_retinopathy
            patient.has_neuropathy = data.complications.has_neuropathy
            patient.has_cardiovascular = data.complications.has_cardiovascular
            patient.has_hypertension = data.complications.has_hypertension
            patient.has_hyperlipidemia = data.complications.has_hyperlipidemia
            patient.has_obesity = data.complications.has_obesity
        
        self.db.add(patient)
        await self.db.commit()
        await self.db.refresh(patient)
        
        logger.info(f"Created diabetic patient: {patient.patient_id}")
        return patient
    
    async def get_patient(self, patient_id: str) -> Optional[DiabeticPatient]:
        """Get patient by ID."""
        result = await self.db.execute(
            select(DiabeticPatient)
            .options(selectinload(DiabeticPatient.medications))
            .where(DiabeticPatient.patient_id == patient_id)
        )
        return result.scalar_one_or_none()
    
    async def update_patient(
        self, patient_id: str, data: DiabeticPatientUpdate
    ) -> Optional[DiabeticPatient]:
        """Update patient profile."""
        patient = await self.get_patient(patient_id)
        if not patient:
            return None
        
        update_data = data.model_dump(exclude_unset=True)
        
        # Handle nested labs
        if "labs" in update_data and update_data["labs"]:
            labs = update_data.pop("labs")
            for key, value in labs.items():
                if value is not None:
                    setattr(patient, key, value)
        
        # Handle nested complications
        if "complications" in update_data and update_data["complications"]:
            complications = update_data.pop("complications")
            for key, value in complications.items():
                setattr(patient, key, value)
        
        # Handle lists
        if "allergies" in update_data:
            patient.allergies = json.dumps(update_data.pop("allergies"))
        if "comorbidities" in update_data:
            patient.comorbidities = json.dumps(update_data.pop("comorbidities"))
        
        # Apply remaining updates
        for key, value in update_data.items():
            if value is not None:
                setattr(patient, key, value)
        
        patient.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(patient)
        
        return patient
    
    async def delete_patient(self, patient_id: str) -> bool:
        """Delete a patient and all related data."""
        patient = await self.get_patient(patient_id)
        if not patient:
            return False
        
        await self.db.delete(patient)
        await self.db.commit()
        return True
    
    async def list_patients(
        self, limit: int = 50, offset: int = 0
    ) -> Tuple[List[DiabeticPatient], int]:
        """List all patients with pagination."""
        # Get total count
        count_result = await self.db.execute(select(func.count(DiabeticPatient.id)))
        total = count_result.scalar()
        
        # Get patients
        result = await self.db.execute(
            select(DiabeticPatient)
            .order_by(DiabeticPatient.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        patients = result.scalars().all()
        
        return list(patients), total
    
    # ==================== Medication Management ====================
    
    async def add_medication(
        self, patient_id: str, data: MedicationCreate
    ) -> Optional[DiabeticMedication]:
        """Add medication to patient."""
        patient = await self.get_patient(patient_id)
        if not patient:
            return None
        
        medication = DiabeticMedication(
            patient_id=patient.id,
            drug_name=data.drug_name,
            generic_name=data.generic_name,
            drug_class=data.drug_class,
            dose=data.dose,
            frequency=data.frequency,
            route=data.route,
            indication=data.indication,
            is_diabetes_medication=data.is_diabetes_medication,
            start_date=datetime.utcnow()
        )
        
        self.db.add(medication)
        await self.db.commit()
        await self.db.refresh(medication)
        
        return medication
    
    async def remove_medication(self, patient_id: str, medication_id: int) -> bool:
        """Remove medication from patient."""
        patient = await self.get_patient(patient_id)
        if not patient:
            return False
        
        result = await self.db.execute(
            delete(DiabeticMedication).where(
                DiabeticMedication.id == medication_id,
                DiabeticMedication.patient_id == patient.id
            )
        )
        await self.db.commit()
        return result.rowcount > 0
    
    async def get_patient_medications(
        self, patient_id: str, active_only: bool = True
    ) -> List[DiabeticMedication]:
        """Get all medications for a patient."""
        patient = await self.get_patient(patient_id)
        if not patient:
            return []
        
        query = select(DiabeticMedication).where(
            DiabeticMedication.patient_id == patient.id
        )
        if active_only:
            query = query.where(DiabeticMedication.is_active == True)
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    # ==================== Risk Assessment ====================
    
    async def check_drug_risk(
        self, patient_id: str, drug_name: str
    ) -> Optional[DrugRiskCheckResponse]:
        """Check the risk of a drug for a specific patient."""
        patient = await self.get_patient(patient_id)
        if not patient:
            return None
        
        # Get current medications
        medications = await self.get_patient_medications(patient_id)
        current_meds = [m.drug_name for m in medications]
        
        # Build patient context dict
        patient_context = self._build_patient_context(patient)
        
        # RULES ARE PRIMARY - clinically validated logic
        # ML is SUPPLEMENTARY only - for additional insights
        
        # Assess risk using rule-based system (PRIMARY)
        assessment = self.rules.assess_drug_risk(drug_name, patient_context, current_meds)
        
        # Save assessment to database
        await self._save_risk_assessment(patient, drug_name, assessment)
        
        # Structured log of rule hit
        logger.info({
            "event": "diabetic_rule_hit",
            "drug": drug_name,
            "risk_level": assessment.risk_level,
            "severity": assessment.severity,
            "risk_score": assessment.risk_score,
            "rule_refs": assessment.rule_references,
            "evidence_sources": assessment.evidence_sources,
            "patient_factors": assessment.patient_factors,
            "patient_id": patient.patient_id,
        })

        # Build response from RULES (always primary)
        response = self._assessment_to_response(assessment)
        decision_source = "rules_primary"

        # Try ML prediction for supplementary info only
        ml_result = None
        if self.ml_predictor and self.ml_predictor.is_loaded:
            try:
                ml_result = self.ml_predictor.predict(drug_name, patient_context)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Diabetic ML prediction failed: {exc}")

        if ml_result:
            # ML is supplementary - never overrides rules
            # Only add ML info for transparency
            response.ml_risk_level = RiskLevelEnum(ml_result.risk_level)
            response.ml_probability = ml_result.probability
            response.ml_decision_source = decision_source
            response.ml_model_version = ml_result.model_version
            
            # Log if ML disagrees with rules (for monitoring/retraining)
            if ml_result.risk_level != assessment.risk_level:
                logger.warning({
                    "event": "ml_rule_disagreement",
                    "drug": drug_name,
                    "rule_risk": assessment.risk_level,
                    "ml_risk": ml_result.risk_level,
                    "patient_id": patient.patient_id,
                    "patient_factors": {
                        "egfr": patient_context.get("egfr"),
                        "potassium": patient_context.get("potassium"),
                        "age": patient_context.get("age"),
                    }
                })

        return response
    
    async def check_all_medications(
        self, patient_id: str, medications: List[str] = None
    ) -> Optional[MedicationListCheckResponse]:
        """Check all medications for a patient."""
        patient = await self.get_patient(patient_id)
        if not patient:
            return None
        
        # Use provided medications or get from patient profile
        if medications is None:
            patient_meds = await self.get_patient_medications(patient_id)
            medications = [m.drug_name for m in patient_meds]
        
        if not medications:
            return MedicationListCheckResponse(
                patient_id=patient_id,
                total_medications=0,
                safe_count=0,
                caution_count=0,
                high_risk_count=0,
                contraindicated_count=0,
                fatal_count=0,
                assessments=[],
                overall_risk_level="safe",
                critical_alerts=[],
                recommendations=["No medications to assess"]
            )
        
        # Build patient context
        patient_context = self._build_patient_context(patient)
        
        # Assess all medications
        assessments = self.rules.check_medication_list(medications, patient_context)
        
        # Count by risk level
        counts = {"safe": 0, "caution": 0, "high_risk": 0, "contraindicated": 0, "fatal": 0}
        critical_alerts = []
        
        for a in assessments:
            counts[a.risk_level] = counts.get(a.risk_level, 0) + 1
            if a.risk_level in ["fatal", "contraindicated"]:
                critical_alerts.append(f"WARNING: {a.drug_name}: {a.recommendation}")
        
        # Determine overall risk
        if counts["fatal"] > 0:
            overall_risk = "fatal"
        elif counts["contraindicated"] > 0:
            overall_risk = "contraindicated"
        elif counts["high_risk"] > 0:
            overall_risk = "high_risk"
        elif counts["caution"] > 0:
            overall_risk = "caution"
        else:
            overall_risk = "safe"
        
        return MedicationListCheckResponse(
            patient_id=patient_id,
            total_medications=len(medications),
            safe_count=counts["safe"],
            caution_count=counts["caution"],
            high_risk_count=counts["high_risk"],
            contraindicated_count=counts["contraindicated"],
            fatal_count=counts["fatal"],
            assessments=[self._assessment_to_response(a) for a in assessments],
            overall_risk_level=overall_risk,
            critical_alerts=critical_alerts,
            recommendations=self._generate_recommendations(assessments)
        )
    
    async def find_safe_alternatives(
        self, patient_id: str, drug_name: str
    ) -> Optional[SafeAlternativesResponse]:
        """Find safer alternatives for a drug."""
        patient = await self.get_patient(patient_id)
        if not patient:
            return None
        
        # Get current medications
        medications = await self.get_patient_medications(patient_id)
        current_meds = [m.drug_name for m in medications]
        
        # Build patient context
        patient_context = self._build_patient_context(patient)
        
        # Check original drug
        original_assessment = self.rules.assess_drug_risk(drug_name, patient_context, current_meds)
        
        # Find alternatives
        alternatives = self.rules.find_safe_alternatives(drug_name, patient_context, current_meds)
        
        return SafeAlternativesResponse(
            original_drug=drug_name,
            original_risk_level=original_assessment.risk_level,
            alternatives=[
                SafeAlternativeResponse(
                    drug=a["drug"],
                    risk_level=a["risk_level"],
                    risk_score=a["risk_score"],
                    considerations=a["considerations"]
                )
                for a in alternatives
            ]
        )
    
    async def generate_patient_report(
        self, patient_id: str, include_alternatives: bool = True
    ) -> Optional[PatientDDIReportResponse]:
        """Generate a full DDI report for a patient."""
        patient = await self.get_patient(patient_id)
        if not patient:
            return None
        
        # Get medications
        medications = await self.get_patient_medications(patient_id)
        
        # Check all medications
        check_result = await self.check_all_medications(
            patient_id, [m.drug_name for m in medications]
        )
        
        # Find alternatives for risky drugs
        alternatives = {}
        if include_alternatives:
            for assessment in check_result.assessments:
                if assessment.risk_level in ["high_risk", "contraindicated", "fatal"]:
                    alts = await self.find_safe_alternatives(patient_id, assessment.drug_name)
                    if alts:
                        alternatives[assessment.drug_name] = alts.alternatives
        
        # Build monitoring plan
        monitoring = set()
        for assessment in check_result.assessments:
            monitoring.update(assessment.monitoring)
        
        # Calculate safety score (0-100, higher is safer)
        total_score = sum(a.risk_score for a in check_result.assessments)
        max_score = len(check_result.assessments) * 100
        safety_score = 100 - (total_score / max_score * 100) if max_score > 0 else 100
        
        return PatientDDIReportResponse(
            patient=self._patient_to_response(patient),
            report_generated_at=datetime.utcnow(),
            current_medications=[
                MedicationResponse.model_validate(m) for m in medications
            ],
            medication_assessments=check_result.assessments,
            fatal_risks=[
                {"drug": a.drug_name, "reason": a.recommendation}
                for a in check_result.assessments if a.risk_level == "fatal"
            ],
            contraindicated_drugs=[
                {"drug": a.drug_name, "reason": a.recommendation}
                for a in check_result.assessments if a.risk_level == "contraindicated"
            ],
            high_risk_drugs=[
                {"drug": a.drug_name, "reason": a.recommendation, "monitoring": a.monitoring}
                for a in check_result.assessments if a.risk_level == "high_risk"
            ],
            recommended_alternatives=alternatives,
            monitoring_plan=list(monitoring),
            overall_safety_score=round(safety_score, 1),
            action_required=check_result.fatal_count > 0 or check_result.contraindicated_count > 0,
            summary=self._generate_summary(check_result)
        )
    
    # ==================== Helper Methods ====================
    
    def _build_patient_context(self, patient: DiabeticPatient) -> Dict:
        """Build patient context dict for rules engine."""
        return {
            "diabetes_type": patient.diabetes_type,
            "years_with_diabetes": patient.years_with_diabetes,
            "age": patient.age,
            "hba1c": patient.hba1c,
            "fasting_glucose": patient.fasting_glucose,
            "egfr": patient.egfr,
            "creatinine": patient.creatinine,
            "potassium": patient.potassium,
            "alt": patient.alt,
            "ast": patient.ast,
            "has_nephropathy": patient.has_nephropathy,
            "has_retinopathy": patient.has_retinopathy,
            "has_neuropathy": patient.has_neuropathy,
            "has_cardiovascular": patient.has_cardiovascular,
            "has_hypertension": patient.has_hypertension,
            "has_hyperlipidemia": patient.has_hyperlipidemia,
            "has_obesity": patient.has_obesity,
            "bmi": patient.bmi,
        }
    
    def _assessment_to_response(self, assessment: RiskAssessment) -> DrugRiskCheckResponse:
        """Convert RiskAssessment to response schema."""
        return DrugRiskCheckResponse(
            drug_name=assessment.drug_name,
            risk_level=RiskLevelEnum(assessment.risk_level),
            risk_score=assessment.risk_score,
            severity=assessment.severity,
            risk_factors=assessment.risk_factors,
            rule_references=assessment.rule_references,
            evidence_sources=assessment.evidence_sources,
            patient_factors=assessment.patient_factors,
            recommendation=assessment.recommendation,
            alternatives=assessment.alternatives,
            monitoring=assessment.monitoring,
            interactions=assessment.interactions,
            is_safe=assessment.risk_level == "safe",
            is_fatal=assessment.risk_level == "fatal",
            requires_monitoring=len(assessment.monitoring) > 0
        )
    
    def _patient_to_response(self, patient: DiabeticPatient) -> DiabeticPatientResponse:
        """Convert patient model to response schema."""
        return DiabeticPatientResponse(
            id=patient.id,
            patient_id=patient.patient_id,
            name=patient.name,
            age=patient.age,
            gender=patient.gender,
            weight_kg=patient.weight_kg,
            height_cm=patient.height_cm,
            bmi=patient.bmi,
            diabetes_type=patient.diabetes_type,
            years_with_diabetes=patient.years_with_diabetes,
            hba1c=patient.hba1c,
            fasting_glucose=patient.fasting_glucose,
            egfr=patient.egfr,
            kidney_stage=patient.kidney_stage,
            creatinine=patient.creatinine,
            potassium=patient.potassium,
            alt=patient.alt,
            ast=patient.ast,
            has_nephropathy=patient.has_nephropathy,
            has_retinopathy=patient.has_retinopathy,
            has_neuropathy=patient.has_neuropathy,
            has_cardiovascular=patient.has_cardiovascular,
            has_hypertension=patient.has_hypertension,
            has_hyperlipidemia=patient.has_hyperlipidemia,
            has_obesity=patient.has_obesity,
            allergies=json.loads(patient.allergies) if patient.allergies else None,
            comorbidities=json.loads(patient.comorbidities) if patient.comorbidities else None,
            created_at=patient.created_at,
            updated_at=patient.updated_at
        )
    
    async def _save_risk_assessment(
        self, patient: DiabeticPatient, drug_name: str, assessment: RiskAssessment
    ):
        """Save risk assessment to database."""
        risk_record = DiabeticDrugRisk(
            patient_id=patient.id,
            drug_name=drug_name,
            risk_level=assessment.risk_level,
            risk_score=assessment.risk_score,
            risk_factors=json.dumps(assessment.risk_factors),
            recommendation=assessment.recommendation,
            alternative_drugs=json.dumps(assessment.alternatives),
            monitoring_required=json.dumps(assessment.monitoring),
            labs_at_assessment=json.dumps({
                "hba1c": patient.hba1c,
                "egfr": patient.egfr,
                "potassium": patient.potassium,
            })
        )
        self.db.add(risk_record)
        await self.db.commit()
    
    def _generate_recommendations(self, assessments: List[RiskAssessment]) -> List[str]:
        """Generate overall recommendations."""
        recommendations = []
        
        fatal = [a for a in assessments if a.risk_level == "fatal"]
        if fatal:
            recommendations.append(f"🚨 IMMEDIATE ACTION: Stop {', '.join(a.drug_name for a in fatal)}")
        
        contraindicated = [a for a in assessments if a.risk_level == "contraindicated"]
        if contraindicated:
            recommendations.append(f"WARNING: Review and replace: {', '.join(a.drug_name for a in contraindicated)}")
        
        high_risk = [a for a in assessments if a.risk_level == "high_risk"]
        if high_risk:
            recommendations.append(f"Increase monitoring for: {', '.join(a.drug_name for a in high_risk)}")
        
        if not recommendations:
            recommendations.append("Current medication regimen appears safe for this patient")
        
        return recommendations
    
    def _generate_summary(self, result: MedicationListCheckResponse) -> str:
        """Generate summary text."""
        if result.fatal_count > 0:
            return f"CRITICAL: {result.fatal_count} medication(s) pose fatal risk. Immediate review required."
        elif result.contraindicated_count > 0:
            return f"WARNING: {result.contraindicated_count} medication(s) are contraindicated. Review with physician."
        elif result.high_risk_count > 0:
            return f"CAUTION: {result.high_risk_count} medication(s) require close monitoring."
        elif result.caution_count > 0:
            return f"NOTICE: {result.caution_count} medication(s) require standard monitoring."
        else:
            return "All medications appear safe for this diabetic patient."


def create_diabetic_service(db: AsyncSession) -> DiabeticDDIService:
    """Factory function to create diabetic DDI service."""
    return DiabeticDDIService(db)

