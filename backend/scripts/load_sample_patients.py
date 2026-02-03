"""
Load sample patients from JSON into the database via the API.

This script loads the real patient data extracted from lab reports
into the DrugGuard system for DDI analysis testing.

Usage:
    python scripts/load_sample_patients.py
"""

import json
import asyncio
import httpx
from pathlib import Path


API_URL = "http://127.0.0.1:8000"


async def load_sample_patients():
    """Load all sample patients from JSON file."""
    # Load the JSON file
    data_file = Path(__file__).parent.parent / "data" / "sample_patients.json"
    
    if not data_file.exists():
        print(f"Error: {data_file} not found!")
        return
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    patients = data.get("patients", [])
    print(f"Found {len(patients)} patients to load...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for patient in patients:
            # Prepare the patient data in API format
            patient_data = {
                "patient_id": patient["patient_id"],
                "name": patient["name"],
                "age": patient["age"],
                "gender": patient["gender"],
                "diabetes_type": patient["diabetes_type"],
                "years_with_diabetes": patient.get("years_with_diabetes"),
                "labs": patient.get("labs", {}),
                "complications": patient.get("complications", {}),
                "allergies": patient.get("allergies", []),
                "comorbidities": patient.get("comorbidities", [])
            }
            
            # Clean up None values in labs
            if patient_data["labs"]:
                patient_data["labs"] = {k: v for k, v in patient_data["labs"].items() if v is not None}
            
            try:
                # First try to delete if exists (to allow reloading)
                await client.delete(f"{API_URL}/diabetic/patients/{patient['patient_id']}")
                
                # Create the patient
                response = await client.post(
                    f"{API_URL}/diabetic/patients",
                    json=patient_data
                )
                
                if response.status_code == 201:
                    print(f"  [OK] Created patient: {patient['name']} ({patient['patient_id']})")
                elif response.status_code == 200:
                    print(f"  [OK] Updated patient: {patient['name']} ({patient['patient_id']})")
                else:
                    print(f"  [ERROR] Failed to create {patient['patient_id']}: {response.status_code}")
                    print(f"    Response: {response.text[:200]}")
            except Exception as e:
                print(f"  [ERROR] Error creating {patient['patient_id']}: {e}")
    
    print(f"\nDone! Loaded {len(patients)} sample patients.")
    print(f"\nYou can now check drug risks for these patients at:")
    print(f"  Web UI: http://localhost:5173")
    print(f"  API: GET {API_URL}/diabetic/patients")


if __name__ == "__main__":
    asyncio.run(load_sample_patients())
