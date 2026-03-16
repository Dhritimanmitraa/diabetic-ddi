"""
Lab Report Analyzer Service using Google Gemini Vision.

Analyzes uploaded lab reports to extract patient data and lab values
for personalized drug-drug interaction analysis.
"""
import base64
import json
import logging
import re
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.gemini_client import get_gemini_client

settings = get_settings()
logger = logging.getLogger(__name__)

# Gemini model for vision tasks
GEMINI_VISION_MODEL = "gemini-2.0-flash"


# ==================== Schemas ====================

class ExtractedPatientInfo(BaseModel):
    """Patient demographics extracted from report."""
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    patient_id: Optional[str] = None
    report_date: Optional[str] = None


class ExtractedGlucoseValues(BaseModel):
    """Glucose-related lab values."""
    hba1c: Optional[float] = Field(None, description="HbA1c percentage")
    fasting_glucose: Optional[float] = Field(None, description="Fasting blood sugar mg/dL")
    postprandial_glucose: Optional[float] = Field(None, description="Post-meal glucose mg/dL")
    mean_blood_glucose: Optional[float] = Field(None, description="Mean blood glucose mg/dL")
    random_glucose: Optional[float] = Field(None, description="Random blood sugar mg/dL")


class ExtractedKidneyValues(BaseModel):
    """Kidney function lab values."""
    creatinine: Optional[float] = Field(None, description="Serum creatinine mg/dL")
    egfr: Optional[float] = Field(None, description="eGFR mL/min/1.73m²")
    urea: Optional[float] = Field(None, description="Blood urea mg/dL")
    bun: Optional[float] = Field(None, description="Blood urea nitrogen mg/dL")
    uric_acid: Optional[float] = Field(None, description="Uric acid mg/dL")


class ExtractedLipidProfile(BaseModel):
    """Lipid profile values."""
    total_cholesterol: Optional[float] = Field(None, description="Total cholesterol mg/dL")
    triglycerides: Optional[float] = Field(None, description="Triglycerides mg/dL")
    hdl_cholesterol: Optional[float] = Field(None, description="HDL cholesterol mg/dL")
    ldl_cholesterol: Optional[float] = Field(None, description="LDL cholesterol mg/dL")
    vldl_cholesterol: Optional[float] = Field(None, description="VLDL cholesterol mg/dL")


class ExtractedLiverValues(BaseModel):
    """Liver function values."""
    alt: Optional[float] = Field(None, description="ALT/SGPT U/L")
    ast: Optional[float] = Field(None, description="AST/SGOT U/L")
    alp: Optional[float] = Field(None, description="Alkaline phosphatase U/L")
    bilirubin: Optional[float] = Field(None, description="Total bilirubin mg/dL")


class ExtractedThyroidValues(BaseModel):
    """Thyroid function values."""
    t3: Optional[float] = Field(None, description="T3 ng/mL")
    t4: Optional[float] = Field(None, description="T4 ug/dL")
    tsh: Optional[float] = Field(None, description="TSH uIU/mL")


class ExtractedLabValues(BaseModel):
    """All extracted lab values from a report."""
    patient: ExtractedPatientInfo = Field(default_factory=ExtractedPatientInfo)
    glucose: ExtractedGlucoseValues = Field(default_factory=ExtractedGlucoseValues)
    kidney: ExtractedKidneyValues = Field(default_factory=ExtractedKidneyValues)
    lipid: ExtractedLipidProfile = Field(default_factory=ExtractedLipidProfile)
    liver: ExtractedLiverValues = Field(default_factory=ExtractedLiverValues)
    thyroid: ExtractedThyroidValues = Field(default_factory=ExtractedThyroidValues)
    potassium: Optional[float] = None
    sodium: Optional[float] = None
    hemoglobin: Optional[float] = None
    other_values: Dict[str, Any] = Field(default_factory=dict)
    raw_text: Optional[str] = None
    extraction_confidence: float = 0.0


class PatientHealthSummary(BaseModel):
    """AI-generated health summary for a patient."""
    overall_status: str = "unknown"  # good, moderate, concerning, critical
    diabetes_control: str = "unknown"
    kidney_function: str = "unknown"
    cardiovascular_risk: str = "unknown"
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ReportAnalysisResult(BaseModel):
    """Complete result of lab report analysis."""
    success: bool = False
    extracted_values: ExtractedLabValues = Field(default_factory=ExtractedLabValues)
    health_summary: PatientHealthSummary = Field(default_factory=PatientHealthSummary)
    model_used: str = ""
    error: Optional[str] = None


# ==================== Extraction Prompt ====================

LAB_EXTRACTION_PROMPT = """Analyze this lab report image and extract ALL values you can find.

Return a JSON object with this EXACT structure (use null for values not found):

{
  "patient": {
    "name": "Patient Name or null",
    "age": 45,
    "gender": "M or F or null",
    "patient_id": "ID if visible or null",
    "report_date": "Date or null"
  },
  "glucose": {
    "hba1c": 6.5,
    "fasting_glucose": 110,
    "postprandial_glucose": 140,
    "mean_blood_glucose": 130,
    "random_glucose": null
  },
  "kidney": {
    "creatinine": 1.0,
    "egfr": 90,
    "urea": null,
    "bun": null,
    "uric_acid": null
  },
  "lipid": {
    "total_cholesterol": 180,
    "triglycerides": 150,
    "hdl_cholesterol": 45,
    "ldl_cholesterol": 100,
    "vldl_cholesterol": 30
  },
  "liver": {
    "alt": 25,
    "ast": 22,
    "alp": null,
    "bilirubin": null
  },
  "thyroid": {
    "t3": 1.2,
    "t4": 7.5,
    "tsh": 3.5
  },
  "potassium": 4.2,
  "sodium": 140,
  "hemoglobin": 14.5,
  "other_values": {}
}

IMPORTANT:
1. Extract NUMERIC values only (no units in the values)
2. Use null for values not found in the report
3. Return ONLY valid JSON, no other text
4. Look for common lab test abbreviations (FBS, PPBS, HbA1c, eGFR, etc.)
"""


HEALTH_SUMMARY_PROMPT = """Based on these lab values for a diabetic patient, provide a health assessment:

Patient: {patient_name}, Age: {age}, Gender: {gender}

Glucose Values:
- HbA1c: {hba1c}% (Target: <7% for diabetics)
- Fasting Glucose: {fasting_glucose} mg/dL (Normal: 70-100)
- Post-meal Glucose: {postprandial_glucose} mg/dL (Normal: <140)

Kidney Function:
- Creatinine: {creatinine} mg/dL (Normal: 0.7-1.2)
- eGFR: {egfr} mL/min (Normal: >90, CKD Stage 3: 30-59)

Lipid Profile:
- Total Cholesterol: {total_cholesterol} mg/dL (Ideal: <200)
- Triglycerides: {triglycerides} mg/dL (Normal: <150)
- HDL: {hdl} mg/dL (Good: >40 men, >50 women)
- LDL: {ldl} mg/dL (Ideal: <100)

Provide a JSON response:
{{
  "overall_status": "good/moderate/concerning/critical",
  "diabetes_control": "well-controlled/fair/poor/uncontrolled",
  "kidney_function": "normal/mildly-impaired/moderately-impaired/severely-impaired",
  "cardiovascular_risk": "low/moderate/high/very-high",
  "key_findings": ["Finding 1", "Finding 2"],
  "recommendations": ["Recommendation 1", "Recommendation 2"],
  "warnings": ["Warning if any critical values"]
}}
"""


PERSONALIZED_DDI_PROMPT = """You are a clinical pharmacist analyzing drug safety for a SPECIFIC patient.

PATIENT PROFILE:
- Name: {patient_name}
- Age: {age}, Gender: {gender}
- Diabetes Type: {diabetes_type}

PATIENT'S ACTUAL LAB VALUES:
- HbA1c: {hba1c}% (Diabetes control)
- Fasting Glucose: {fasting_glucose} mg/dL
- eGFR: {egfr} mL/min/1.73m² (Kidney function)
- Creatinine: {creatinine} mg/dL
- Potassium: {potassium} mEq/L
- Total Cholesterol: {total_cholesterol} mg/dL
- Triglycerides: {triglycerides} mg/dL

DRUG TO ANALYZE: {drug_name}

Provide a PERSONALIZED risk assessment considering THIS patient's specific values:

1. Is this drug SAFE, CAUTION, HIGH-RISK, or CONTRAINDICATED for THIS patient?
2. WHY does this patient's specific lab values affect the drug's safety?
3. What monitoring is needed for THIS patient?
4. What safer alternatives exist for THIS patient's condition?

Be specific about how THEIR values (not general ranges) affect the drug choice.

Return JSON:
{{
  "risk_level": "safe/caution/high_risk/contraindicated",
  "risk_score": 0-100,
  "personalized_reasoning": "Specific explanation based on patient's values",
  "patient_specific_concerns": ["Concern 1 based on their labs", "Concern 2"],
  "monitoring_for_this_patient": ["What to monitor given their values"],
  "alternatives_for_this_patient": ["Safer options for their condition"],
  "dosage_adjustment": "Any dose changes needed for this patient"
}}
"""


# ==================== Main Service ====================

class LabReportAnalyzer:
    """Analyze lab reports using Google Gemini Vision."""
    
    def __init__(self):
        self.gemini_model = None
        self.gemini_available = False
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Gemini Vision model."""
        self.gemini_model = get_gemini_client(GEMINI_VISION_MODEL)
        self.gemini_available = self.gemini_model.is_available
        if self.gemini_available:
            logger.info(f"Lab Report Analyzer initialized with {GEMINI_VISION_MODEL} via {self.gemini_model.sdk}")
        else:
            logger.warning("No Gemini API key found for lab report analysis")
    
    async def analyze_report(
        self, 
        image_data: bytes, 
        filename: str = "report.jpg",
        content_type: str = None
    ) -> ReportAnalysisResult:
        """
        Analyze a lab report image and extract all values.
        
        Args:
            image_data: Raw image bytes
            filename: Original filename
            
        Returns:
            ReportAnalysisResult with extracted values and health summary
        """
        result = ReportAnalysisResult()
        
        if not self.gemini_available:
            result.error = "Gemini Vision not available"
            return result
        
        try:
            # Step 1: Extract lab values
            logger.info(f"Analyzing lab report: {filename}")
            extracted = await self._extract_lab_values(image_data, content_type, filename)
            
            if extracted:
                result.extracted_values = extracted
                result.model_used = GEMINI_VISION_MODEL
                result.success = True
                
                # Step 2: Generate health summary
                summary = await self._generate_health_summary(extracted)
                if summary:
                    result.health_summary = summary
                    
                logger.info(f"Successfully analyzed report, found patient: {extracted.patient.name}")
            else:
                result.error = "Failed to extract values from report"
                
        except Exception as e:
            logger.error(f"Error analyzing lab report: {e}")
            result.error = str(e)
        
        return result
    
    async def _extract_lab_values(self, image_data: bytes, content_type: str = None, filename: str = None) -> Optional[ExtractedLabValues]:
        """Extract lab values from image using Gemini Vision."""
        try:
            # Determine MIME type
            mime_type = content_type
            if not mime_type:
                # Try to detect from filename extension
                if filename:
                    ext = filename.lower().split('.')[-1] if '.' in filename else ''
                    mime_map = {
                        'jpg': 'image/jpeg',
                        'jpeg': 'image/jpeg',
                        'png': 'image/png',
                        'pdf': 'application/pdf',
                        'webp': 'image/webp',
                        'gif': 'image/gif'
                    }
                    mime_type = mime_map.get(ext, 'image/jpeg')
                else:
                    mime_type = 'image/jpeg'
            
            logger.info(f"Processing file with MIME type: {mime_type}")
            
            response = self.gemini_model.generate_with_media(
                LAB_EXTRACTION_PROMPT,
                media_bytes=image_data,
                mime_type=mime_type,
                temperature=0.1,
                max_output_tokens=2000,
            )
            
            # Parse response
            response_text = response.text.strip()
            
            # Clean up JSON if wrapped in markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            data = json.loads(response_text)
            
            # Build ExtractedLabValues
            result = ExtractedLabValues(
                patient=ExtractedPatientInfo(**data.get("patient", {})),
                glucose=ExtractedGlucoseValues(**data.get("glucose", {})),
                kidney=ExtractedKidneyValues(**data.get("kidney", {})),
                lipid=ExtractedLipidProfile(**data.get("lipid", {})),
                liver=ExtractedLiverValues(**data.get("liver", {})),
                thyroid=ExtractedThyroidValues(**data.get("thyroid", {})),
                potassium=data.get("potassium"),
                sodium=data.get("sodium"),
                hemoglobin=data.get("hemoglobin"),
                other_values=data.get("other_values", {}),
                extraction_confidence=0.85  # Gemini generally reliable
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting lab values: {e}")
            return None
    
    async def _generate_health_summary(
        self, 
        values: ExtractedLabValues
    ) -> Optional[PatientHealthSummary]:
        """Generate AI health summary from extracted values."""
        try:
            prompt = HEALTH_SUMMARY_PROMPT.format(
                patient_name=values.patient.name or "Unknown",
                age=values.patient.age or "Unknown",
                gender=values.patient.gender or "Unknown",
                hba1c=values.glucose.hba1c or "N/A",
                fasting_glucose=values.glucose.fasting_glucose or "N/A",
                postprandial_glucose=values.glucose.postprandial_glucose or "N/A",
                creatinine=values.kidney.creatinine or "N/A",
                egfr=values.kidney.egfr or "N/A",
                total_cholesterol=values.lipid.total_cholesterol or "N/A",
                triglycerides=values.lipid.triglycerides or "N/A",
                hdl=values.lipid.hdl_cholesterol or "N/A",
                ldl=values.lipid.ldl_cholesterol or "N/A"
            )
            
            response = self.gemini_model.generate_text(
                prompt,
                temperature=0.2,
                max_output_tokens=1000,
            )
            
            response_text = response.text.strip()
            
            # Clean JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response_text)
            return PatientHealthSummary(**data)
            
        except Exception as e:
            logger.error(f"Error generating health summary: {e}")
            return None
    
    async def get_personalized_ddi_analysis(
        self,
        patient_data: Dict[str, Any],
        drug_name: str
    ) -> Dict[str, Any]:
        """
        Get personalized DDI analysis for a specific patient and drug.
        
        Args:
            patient_data: Patient's lab values and demographics
            drug_name: Drug to analyze
            
        Returns:
            Personalized risk assessment
        """
        if not self.gemini_available:
            return {"error": "Gemini not available"}
        
        try:
            prompt = PERSONALIZED_DDI_PROMPT.format(
                patient_name=patient_data.get("name", "Unknown"),
                age=patient_data.get("age", "Unknown"),
                gender=patient_data.get("gender", "Unknown"),
                diabetes_type=patient_data.get("diabetes_type", "type_2"),
                hba1c=patient_data.get("hba1c", "N/A"),
                fasting_glucose=patient_data.get("fasting_glucose", "N/A"),
                egfr=patient_data.get("egfr", "N/A"),
                creatinine=patient_data.get("creatinine", "N/A"),
                potassium=patient_data.get("potassium", "N/A"),
                total_cholesterol=patient_data.get("total_cholesterol", "N/A"),
                triglycerides=patient_data.get("triglycerides", "N/A"),
                drug_name=drug_name
            )
            
            response = self.gemini_model.generate_text(
                prompt,
                temperature=0.3,
                max_output_tokens=1500,
            )
            
            response_text = response.text.strip()
            
            # Clean JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"Error getting personalized DDI: {e}")
            return {"error": str(e)}


# ==================== Singleton ====================

_analyzer_instance = None


def get_lab_report_analyzer() -> LabReportAnalyzer:
    """Get singleton instance of LabReportAnalyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = LabReportAnalyzer()
    return _analyzer_instance
