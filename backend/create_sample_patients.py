"""Create sample patients for demonstration."""

import requests

API = "http://localhost:8000"

patients = [
    {
        "patient_id": "DM001",
        "name": "John Smith",
        "age": 72,
        "gender": "M",
        "diabetes_type": "type_2",
        "years_with_diabetes": 15,
        "labs": {
            "hba1c": 8.2,
            "fasting_glucose": 180,
            "egfr": 25,
            "creatinine": 3.2,
            "potassium": 5.5,
        },
        "complications": {
            "has_nephropathy": True,
            "has_retinopathy": True,
            "has_neuropathy": True,
            "has_cardiovascular": True,
            "has_hypertension": True,
        },
    },
    {
        "patient_id": "DM002",
        "name": "Sarah Johnson",
        "age": 45,
        "gender": "F",
        "diabetes_type": "type_2",
        "years_with_diabetes": 5,
        "labs": {
            "hba1c": 7.1,
            "fasting_glucose": 130,
            "egfr": 85,
            "creatinine": 0.9,
            "potassium": 4.2,
        },
        "complications": {
            "has_nephropathy": False,
            "has_retinopathy": False,
            "has_neuropathy": False,
            "has_cardiovascular": False,
            "has_hypertension": True,
        },
    },
    {
        "patient_id": "DM003",
        "name": "Michael Brown",
        "age": 65,
        "gender": "M",
        "diabetes_type": "type_2",
        "years_with_diabetes": 10,
        "labs": {
            "hba1c": 7.8,
            "fasting_glucose": 160,
            "egfr": 45,
            "creatinine": 1.8,
            "potassium": 5.1,
            "alt": 85,
            "ast": 78,
        },
        "complications": {
            "has_nephropathy": True,
            "has_retinopathy": False,
            "has_neuropathy": True,
            "has_cardiovascular": True,
            "has_hypertension": True,
            "has_hyperlipidemia": True,
        },
    },
    {
        "patient_id": "DM004",
        "name": "Emily Davis",
        "age": 32,
        "gender": "F",
        "diabetes_type": "type_1",
        "years_with_diabetes": 20,
        "labs": {
            "hba1c": 6.8,
            "fasting_glucose": 110,
            "egfr": 95,
            "creatinine": 0.7,
            "potassium": 4.0,
        },
        "complications": {
            "has_nephropathy": False,
            "has_retinopathy": True,
            "has_neuropathy": False,
            "has_cardiovascular": False,
            "has_hypertension": False,
        },
    },
]

print("Creating sample patients...")
for p in patients:
    try:
        r = requests.post(f"{API}/diabetic/patients", json=p)
        if r.status_code in [200, 201]:
            print(f'  Created: {p["name"]} ({p["patient_id"]})')
        else:
            print(f'  {p["patient_id"]}: {r.status_code}')
    except Exception as e:
        print(f"  Error: {e}")

print("\nPatient profiles:")
print(
    "  DM001 - Severe kidney disease (eGFR 25), high potassium - METFORMIN CONTRAINDICATED"
)
print("  DM002 - Healthy kidneys (eGFR 85), normal labs - Most drugs SAFE")
print("  DM003 - Moderate kidney disease, elevated liver - DOSE ADJUSTMENTS needed")
print("  DM004 - Type 1, good control, retinopathy only - Most drugs SAFE")
