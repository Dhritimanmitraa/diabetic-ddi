"""
Seed demo diabetic patients for testing and demonstration.

Creates sample patients with various conditions to showcase the system.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import init_db, async_session
from app.diabetic.models import DiabeticPatient, DiabeticMedication
from app.diabetic.schemas import DiabeticPatientCreate, MedicationCreate
from app.diabetic.service import DiabeticDDIService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEMO_PATIENTS = [
    {
        "patient_id": "DEMO001",
        "name": "John Smith",
        "age": 65,
        "gender": "male",
        "weight_kg": 85,
        "height_cm": 175,
        "diabetes_type": "type_2",
        "years_with_diabetes": 12,
        "labs": {
            "hba1c": 7.2,
            "fasting_glucose": 145,
            "egfr": 45,
            "creatinine": 1.8,
            "potassium": 4.2,
        },
        "complications": {
            "has_nephropathy": True,
            "has_cardiovascular": True,
            "has_neuropathy": False,
            "has_retinopathy": False,
            "has_hypertension": True,
            "has_hyperlipidemia": True,
        },
        "medications": [
            {"drug_name": "Metformin", "dosage": "1000mg", "frequency": "twice daily"},
            {"drug_name": "Lisinopril", "dosage": "10mg", "frequency": "once daily"},
            {"drug_name": "Atorvastatin", "dosage": "20mg", "frequency": "once daily"},
        ],
    },
    {
        "patient_id": "DEMO002",
        "name": "Sarah Johnson",
        "age": 52,
        "gender": "female",
        "weight_kg": 72,
        "height_cm": 162,
        "diabetes_type": "type_2",
        "years_with_diabetes": 8,
        "labs": {
            "hba1c": 6.8,
            "fasting_glucose": 128,
            "egfr": 78,
            "creatinine": 1.1,
            "potassium": 3.9,
        },
        "complications": {
            "has_nephropathy": False,
            "has_cardiovascular": False,
            "has_neuropathy": True,
            "has_retinopathy": False,
            "has_hypertension": False,
            "has_hyperlipidemia": True,
        },
        "medications": [
            {"drug_name": "Glipizide", "dosage": "5mg", "frequency": "twice daily"},
            {
                "drug_name": "Gabapentin",
                "dosage": "300mg",
                "frequency": "three times daily",
            },
        ],
    },
    {
        "patient_id": "DEMO003",
        "name": "Michael Chen",
        "age": 58,
        "gender": "male",
        "weight_kg": 90,
        "height_cm": 180,
        "diabetes_type": "type_2",
        "years_with_diabetes": 15,
        "labs": {
            "hba1c": 8.5,
            "fasting_glucose": 165,
            "egfr": 28,
            "creatinine": 2.5,
            "potassium": 4.8,
        },
        "complications": {
            "has_nephropathy": True,
            "has_cardiovascular": True,
            "has_neuropathy": True,
            "has_retinopathy": True,
            "has_hypertension": True,
            "has_hyperlipidemia": True,
        },
        "medications": [
            {
                "drug_name": "Insulin Glargine",
                "dosage": "30 units",
                "frequency": "once daily",
            },
            {"drug_name": "Furosemide", "dosage": "40mg", "frequency": "once daily"},
            {"drug_name": "Amlodipine", "dosage": "5mg", "frequency": "once daily"},
        ],
    },
    {
        "patient_id": "DEMO004",
        "name": "Emily Davis",
        "age": 45,
        "gender": "female",
        "weight_kg": 68,
        "height_cm": 165,
        "diabetes_type": "type_1",
        "years_with_diabetes": 20,
        "labs": {
            "hba1c": 7.0,
            "fasting_glucose": 135,
            "egfr": 92,
            "creatinine": 0.9,
            "potassium": 4.0,
        },
        "complications": {
            "has_nephropathy": False,
            "has_cardiovascular": False,
            "has_neuropathy": False,
            "has_retinopathy": True,
            "has_hypertension": False,
            "has_hyperlipidemia": False,
        },
        "medications": [
            {
                "drug_name": "Insulin Lispro",
                "dosage": "variable",
                "frequency": "before meals",
            },
            {
                "drug_name": "Insulin Detemir",
                "dosage": "20 units",
                "frequency": "bedtime",
            },
        ],
    },
    {
        "patient_id": "DEMO005",
        "name": "Robert Williams",
        "age": 70,
        "gender": "male",
        "weight_kg": 88,
        "height_cm": 178,
        "diabetes_type": "type_2",
        "years_with_diabetes": 18,
        "labs": {
            "hba1c": 7.8,
            "fasting_glucose": 152,
            "egfr": 35,
            "creatinine": 2.0,
            "potassium": 5.1,
        },
        "complications": {
            "has_nephropathy": True,
            "has_cardiovascular": True,
            "has_neuropathy": True,
            "has_retinopathy": False,
            "has_hypertension": True,
            "has_hyperlipidemia": True,
        },
        "medications": [
            {"drug_name": "Metformin", "dosage": "500mg", "frequency": "twice daily"},
            {"drug_name": "Glyburide", "dosage": "5mg", "frequency": "twice daily"},
            {"drug_name": "Verapamil", "dosage": "120mg", "frequency": "twice daily"},
            {"drug_name": "Warfarin", "dosage": "5mg", "frequency": "once daily"},
        ],
    },
]


async def seed_demo_patients():
    """Create demo patients in the database."""
    await init_db()

    async with async_session() as db:
        service = DiabeticDDIService(db)

        created_count = 0
        skipped_count = 0

        for patient_data in DEMO_PATIENTS:
            patient_id = patient_data["patient_id"]

            # Extract medications before creating patient
            medications = patient_data.pop("medications", [])

            # Check if patient already exists
            existing = await service.get_patient(patient_id)
            if existing:
                logger.info(
                    f"Patient {patient_id} already exists, adding medications..."
                )
                skipped_count += 1
                patient = existing
            else:
                # Create patient
                try:
                    patient_create = DiabeticPatientCreate(**patient_data)
                    patient = await service.create_patient(patient_create)
                    logger.info(f"Created patient: {patient.name} ({patient_id})")
                    created_count += 1
                except Exception as e:
                    logger.error(f"Failed to create patient {patient_id}: {e}")
                    continue

            # Add medications (whether patient is new or existing)
            for med in medications:
                try:
                    med_name = med.get("drug_name", med.get("name", ""))

                    # Try to add medication (will fail if duplicate)
                    result = await service.add_medication(
                        patient_id, MedicationCreate(**med)
                    )
                    if result:
                        logger.info(f"  → Added medication: {med_name}")
                    else:
                        logger.info(
                            f"  → Medication {med_name} already exists or failed to add"
                        )
                except Exception as e:
                    # If it's a duplicate error, that's okay
                    error_msg = str(e).lower()
                    if (
                        "duplicate" in error_msg
                        or "already exists" in error_msg
                        or "unique" in error_msg
                    ):
                        logger.info(
                            f"  → Medication {med_name} already exists, skipping..."
                        )
                    else:
                        logger.warning(f"  → Failed to add {med_name}: {e}")

        await db.commit()

        print("\n" + "=" * 60)
        print("DEMO PATIENTS SEEDING COMPLETE")
        print("=" * 60)
        print(f"Created: {created_count} patients")
        print(f"Skipped: {skipped_count} patients (already exist)")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(seed_demo_patients())
