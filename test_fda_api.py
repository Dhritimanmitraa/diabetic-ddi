"""
Simple script to test OpenFDA API
Run this with: python test_fda_api.py
"""
import urllib.request
import json

print("=" * 60)
print("Testing OpenFDA API")
print("=" * 60)

# Test 1: Get drug labels
print("\n1. Testing Drug Labels endpoint...")
url = "https://api.fda.gov/drug/label.json?limit=3"
print(f"   URL: {url}")

try:
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read())
        total = data.get('meta', {}).get('results', {}).get('total', 0)
        results = data.get('results', [])
        
        print(f"   [OK] Success! Found {total:,} total drugs in database")
        print(f"   [OK] Retrieved {len(results)} drugs in this response")
        
        if results:
            first_drug = results[0]
            openfda = first_drug.get('openfda', {})
            brand = openfda.get('brand_name', ['Unknown'])[0] if openfda.get('brand_name') else 'Unknown'
            generic = openfda.get('generic_name', ['Unknown'])[0] if openfda.get('generic_name') else 'Unknown'
            print(f"   [DRUG] Example drug: {brand} (Generic: {generic})")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# Test 2: Get adverse events
print("\n2. Testing Adverse Events endpoint...")
url = "https://api.fda.gov/drug/event.json?limit=2"
print(f"   URL: {url}")

try:
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read())
        total = data.get('meta', {}).get('results', {}).get('total', 0)
        results = data.get('results', [])
        
        print(f"   [OK] Success! Found {total:,} total adverse events in database")
        print(f"   [OK] Retrieved {len(results)} events in this response")
        
        if results:
            first_event = results[0]
            patient = first_event.get('patient', {})
            drugs = patient.get('drug', [])
            if drugs:
                drug_names = [d.get('medicinalproduct', 'Unknown') for d in drugs[:2]]
                print(f"   [EVENT] Example event with drugs: {', '.join(drug_names)}")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
print("\n[TIP] You can also test in your browser:")
print("   https://api.fda.gov/drug/label.json?limit=5")
print("   https://api.fda.gov/drug/event.json?limit=5")

