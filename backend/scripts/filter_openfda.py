"""
Filter OpenFDA/RxNorm Drug Interaction Data for Diabetes Relevance.

Filters the 569k+ line real_drug_data.json to extract only:
1. Diabetes-relevant adverse effects
2. Aggregated signal strength from multiple reports
3. Capped influence for safety

Output: openfda_diabetes_filtered.json
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

BASE_DIR = Path(__file__).parent.parent  # backend/
DATA_DIR = BASE_DIR / "data"
INPUT_PATH = DATA_DIR / "real_drug_data.json"
OUTPUT_PATH = DATA_DIR / "openfda_diabetes_filtered.json"

# Effects relevant to diabetic patients
DIABETES_RELEVANT_EFFECTS = {
    # Glucose-related
    "hyperglycemia", "hypoglycemia", "diabetic ketoacidosis", "dka",
    "blood glucose increased", "blood glucose decreased", "glucose tolerance impaired",
    "hyperglycaemia", "hypoglycaemia",
    
    # Kidney-related (important for diabetic nephropathy)
    "renal failure", "renal impairment", "acute kidney injury", "kidney failure",
    "creatinine increased", "renal failure acute", "nephropathy",
    
    # Cardiovascular (common comorbidity)
    "cardiac arrest", "sudden cardiac death", "heart failure", "cardiac failure",
    "myocardial infarction", "arrhythmia", "qt prolongation",
    
    # Electrolyte disturbances
    "hyperkalaemia", "hyperkalemia", "hypokalaemia", "hypokalemia",
    "hyponatremia", "hyponatraemia",
    
    # Metabolic
    "lactic acidosis", "metabolic acidosis", "weight increased", "weight gain",
    
    # Other diabetes-relevant
    "pancreatitis", "hepatotoxicity", "liver failure", "hepatic failure",
}

# Minimum confidence to include
MIN_CONFIDENCE = 0.5

# Maximum risk adjustment
MAX_RISK_ADJUSTMENT = 15


def normalize_effect(effect: str) -> Set[str]:
    """Extract relevant effects from a potentially multi-effect string."""
    effect_lower = effect.lower()
    found = set()
    
    for relevant in DIABETES_RELEVANT_EFFECTS:
        if relevant in effect_lower:
            found.add(relevant)
    
    return found


def aggregate_signals(interactions: List[Dict]) -> Dict:
    """Aggregate interaction signals by drug pair."""
    
    aggregated = defaultdict(lambda: {
        "count": 0,
        "confidence_sum": 0.0,
        "effects": set(),
        "severities": [],
    })
    
    filtered_count = 0
    relevant_count = 0
    
    for interaction in interactions:
        drug1 = interaction.get("drug1_name", "").upper()
        drug2 = interaction.get("drug2_name", "").upper()
        effect = interaction.get("effect", "")
        confidence = interaction.get("confidence_score", 0)
        severity = interaction.get("severity", "")
        
        # Skip low confidence
        if confidence < MIN_CONFIDENCE:
            filtered_count += 1
            continue
        
        # Check relevance
        relevant_effects = normalize_effect(effect)
        if not relevant_effects:
            filtered_count += 1
            continue
        
        relevant_count += 1
        
        # Aggregate
        key = tuple(sorted([drug1, drug2]))
        aggregated[key]["count"] += 1
        aggregated[key]["confidence_sum"] += confidence
        aggregated[key]["effects"].update(relevant_effects)
        aggregated[key]["severities"].append(severity)
    
    print(f"Filtered {filtered_count} irrelevant interactions")
    print(f"Kept {relevant_count} relevant interactions")
    
    return aggregated


def calculate_risk_adjustment(agg_data: Dict) -> float:
    """Calculate capped risk adjustment from aggregated signals."""
    
    count = agg_data["count"]
    confidence_sum = agg_data["confidence_sum"]
    
    # Base adjustment from signal strength
    raw_adjustment = min(confidence_sum * 2, count * 3)
    
    # Cap the adjustment
    return min(raw_adjustment, MAX_RISK_ADJUSTMENT)


def filter_openfda():
    """Main filtering function."""
    
    print(f"Loading {INPUT_PATH}...")
    
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    interactions = data.get("interactions", [])
    print(f"Total interactions: {len(interactions)}")
    
    # Aggregate by drug pair
    aggregated = aggregate_signals(interactions)
    
    # Build final output
    final_interactions = []
    
    for (drug1, drug2), agg in aggregated.items():
        risk_adj = calculate_risk_adjustment(agg)
        
        # Determine dominant severity
        severities = agg["severities"]
        if severities:
            severity_counts = defaultdict(int)
            for s in severities:
                severity_counts[s.lower()] += 1
            dominant_severity = max(severity_counts, key=severity_counts.get)
        else:
            dominant_severity = "moderate"
        
        final_interactions.append({
            "drug1": drug1,
            "drug2": drug2,
            "report_count": agg["count"],
            "signal_strength": round(agg["confidence_sum"], 2),
            "risk_adjustment": round(risk_adj, 1),
            "effects": list(agg["effects"]),
            "dominant_severity": dominant_severity,
        })
    
    # Sort by risk adjustment
    final_interactions.sort(key=lambda x: x["risk_adjustment"], reverse=True)
    
    output = {
        "metadata": {
            "source": "OpenFDA FAERS",
            "filtered_for": "diabetes_relevance",
            "total_pairs": len(final_interactions),
            "max_risk_adjustment": MAX_RISK_ADJUSTMENT,
        },
        "interactions": final_interactions,
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved {len(final_interactions)} filtered interactions to {OUTPUT_PATH}")
    print(f"Top 5 by risk adjustment:")
    for i in final_interactions[:5]:
        print(f"  {i['drug1']} + {i['drug2']}: adj={i['risk_adjustment']}, effects={i['effects']}")


if __name__ == "__main__":
    filter_openfda()
