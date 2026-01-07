"""Test the patients API endpoint."""

import requests
import json

try:
    response = requests.get("http://localhost:8000/diabetic/patients")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        patients = response.json()
        print(f"\nFound {len(patients)} patients via API:")
        print("=" * 60)
        for p in patients[:10]:  # Show first 10
            print(f"  {p.get('patient_id', 'N/A')}: {p.get('name', 'N/A')}")
        print("=" * 60)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error connecting to API: {e}")
    print("Is the backend server running on port 8000?")
