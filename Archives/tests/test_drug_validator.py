"""
Test the drug validator module.
"""
import sys
sys.path.insert(0, '.')

from app.diabetic.drug_validator import validate_drug_name, is_valid_drug

# Test cases
test_cases = [
    # Invalid drug names (should be rejected)
    ("lol", False),
    ("asdf", False),
    ("hello", False),
    ("test", False),
    ("", False),
    ("a", False),
    ("yo", False),
    
    # Valid drug names (should be accepted)
    ("metformin", True),
    ("aspirin", True),
    ("lisinopril", True),
    ("insulin", True),
    ("atorvastatin", True),
    ("omeprazole", True),
    ("ibuprofen", True),
    ("gabapentin", True),
    
    # Valid by pattern matching (drug-like suffixes)
    ("newdruglol", False),  # lol at end is not a drug suffix
    ("somethingpril", True),  # -pril is ACE inhibitor suffix
    ("mystatin", True),  # -statin is statin suffix
    ("testolol", True),  # -olol is beta blocker suffix
]

print("Testing drug validator...\n")
print("-" * 60)

passed = 0
failed = 0

for drug_name, expected_valid in test_cases:
    is_valid, reason = validate_drug_name(drug_name)
    status = "✅ PASS" if is_valid == expected_valid else "❌ FAIL"
    
    if is_valid == expected_valid:
        passed += 1
    else:
        failed += 1
    
    print(f"{status} | {drug_name:20} | Expected: {expected_valid:5} | Got: {is_valid:5}")
    if reason:
        print(f"       Reason: {reason}")

print("-" * 60)
print(f"\nResults: {passed} passed, {failed} failed")
