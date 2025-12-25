"""
Validation harness for diabetic rules.

Contains curated contraindication/caution cases and a runner that can be used
in nightly regression or unit tests.
"""
from typing import List, Dict, Any

from app.diabetic.rules import DiabeticDrugRules


class ValidationCase:
    def __init__(self, name: str, patient: Dict[str, Any], drug: str, expected_severity: str):
        self.name = name
        self.patient = patient
        self.drug = drug
        self.expected_severity = expected_severity


# Curated cases (expand as needed)
VALIDATION_CASES: List[ValidationCase] = [
    ValidationCase(
        name="Metformin eGFR 25 contraindicated",
        patient={"egfr": 25},
        drug="metformin",
        expected_severity="contraindicated",
    ),
    ValidationCase(
        name="Glyburide eGFR 35 high risk",
        patient={"egfr": 35},
        drug="glyburide",
        expected_severity="major",
    ),
    ValidationCase(
        name="Alcohol hypoglycemia risk",
        patient={},
        drug="alcohol",
        expected_severity="moderate",
    ),
    ValidationCase(
        name="TZD heart failure caution",
        patient={"has_cardiovascular": True},
        drug="pioglitazone",
        expected_severity="major",
    ),
    ValidationCase(
        name="SGLT2 low eGFR caution",
        patient={"egfr": 35},
        drug="empagliflozin",
        expected_severity="moderate",
    ),
]


def run_validation() -> List[Dict[str, Any]]:
    """Run validation cases and return results."""
    rules = DiabeticDrugRules()
    results = []
    for case in VALIDATION_CASES:
        ra = rules.assess_drug_risk(case.drug, case.patient, [])
        results.append({
            "case": case.name,
            "drug": case.drug,
            "expected": case.expected_severity,
            "got": ra.severity,
            "risk_level": ra.risk_level,
            "risk_score": ra.risk_score,
            "pass": ra.severity == case.expected_severity,
            "rule_refs": ra.rule_references,
        })
    return results


if __name__ == "__main__":
    for r in run_validation():
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] {r['case']} -> expected {r['expected']} got {r['got']} (risk_level={r['risk_level']})")

