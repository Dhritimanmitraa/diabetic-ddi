import pytest

from app.diabetic.rules import DiabeticDrugRules
from app.diabetic.validation import VALIDATION_CASES


@pytest.mark.parametrize("case", VALIDATION_CASES)
def test_validation_cases(case):
    rules = DiabeticDrugRules()
    ra = rules.assess_drug_risk(case.drug, case.patient, [])
    assert ra.severity == case.expected_severity, f"{case.name}: expected {case.expected_severity}, got {ra.severity}"


def test_hyperkalemia_combo():
    rules = DiabeticDrugRules()
    patient = {"potassium": 5.2}
    ra = rules.assess_drug_risk("lisinopril", patient, ["spironolactone"])
    assert ra.risk_level in ["high_risk", "contraindicated"]
    assert any("hyperkalemia" in rf.lower() for rf in ra.rule_references)

