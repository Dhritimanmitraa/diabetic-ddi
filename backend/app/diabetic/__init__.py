"""
Diabetic Patient Drug-Drug Interaction Module

This module provides specialized DDI analysis for diabetic patients, including:
- Patient profile management with diabetes-specific parameters
- Drug risk stratification for diabetic patients
- Contraindication detection based on diabetes type, complications, and labs
- Safe alternative suggestions considering diabetes management
"""

from app.diabetic.models import DiabeticPatient, DiabeticDrugRisk, DiabeticMedication
from app.diabetic.rules import DiabeticDrugRules
from app.diabetic.service import DiabeticDDIService

__all__ = [
    "DiabeticPatient",
    "DiabeticDrugRisk", 
    "DiabeticMedication",
    "DiabeticDrugRules",
    "DiabeticDDIService",
]

