"""
Diabetic Drug Rules Engine.

Contains curated rules for drug safety in diabetic patients based on:
- Clinical guidelines (ADA, AACE)
- Drug-specific contraindications
- Lab-based dosing adjustments
- Complication-specific risks
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Path to the diabetes medications JSON file
_DATA_DIR = Path(__file__).parent / "data"
_MEDICATIONS_JSON_PATH = _DATA_DIR / "diabetes_medications.json"


@dataclass
class RiskAssessment:
    """Result of risk assessment for a drug."""

    drug_name: str
    risk_level: str  # safe, caution, high_risk, contraindicated, fatal
    risk_score: float  # 0-100
    severity: str  # minor, moderate, major, contraindicated, fatal
    risk_factors: List[str]
    rule_references: List[str]
    evidence_sources: List[str]
    patient_factors: List[str]
    recommendation: str
    alternatives: List[str]
    monitoring: List[str]
    interactions: List[Dict]


class DiabeticDrugRules:
    """
    Rule engine for assessing drug safety in diabetic patients.

    Uses a combination of:
    1. Hard-coded clinical rules (guidelines-based)
    2. Lab-based thresholds
    3. Complication-specific contraindications
    4. Drug interaction analysis
    """

    # Drug classes that affect blood glucose
    HYPOGLYCEMIA_RISK_DRUGS = {
        "sulfonylureas": ["glimepiride", "glipizide", "glyburide", "glibenclamide"],
        "meglitinides": ["repaglinide", "nateglinide"],
        "insulin": [
            "insulin",
            "insulin glargine",
            "insulin lispro",
            "insulin aspart",
            "insulin detemir",
        ],
    }

    HYPERGLYCEMIA_RISK_DRUGS = {
        "corticosteroids": [
            "prednisone",
            "dexamethasone",
            "hydrocortisone",
            "methylprednisolone",
            "prednisolone",
        ],
        "thiazides": ["hydrochlorothiazide", "chlorthalidone", "indapamide"],
        "atypical_antipsychotics": [
            "olanzapine",
            "clozapine",
            "risperidone",
            "quetiapine",
        ],
        "beta_blockers": ["propranolol", "metoprolol", "atenolol", "carvedilol"],
        "niacin": ["niacin", "nicotinic acid"],
        "protease_inhibitors": ["ritonavir", "lopinavir"],
    }

    # Drugs that mask hypoglycemia symptoms
    MASK_HYPOGLYCEMIA = ["propranolol", "metoprolol", "atenolol", "nadolol", "timolol"]

    # Nephrotoxic drugs (caution with diabetic nephropathy)
    NEPHROTOXIC_DRUGS = [
        "nsaids",
        "ibuprofen",
        "naproxen",
        "diclofenac",
        "celecoxib",
        "aminoglycosides",
        "gentamicin",
        "tobramycin",
        "amikacin",
        "amphotericin b",
        "cisplatin",
        "cyclosporine",
        "tacrolimus",
        "lithium",
        "acyclovir",
        "tenofovir",
    ]

    # Contraindicated based on eGFR thresholds
    EGFR_CONTRAINDICATIONS = {
        # Diabetes medications
        "metformin": {
            "contraindicated_below": 30,
            "caution_below": 45,
            "reason": "Risk of lactic acidosis",
        },
        "glyburide": {
            "contraindicated_below": 30,
            "caution_below": 60,
            "reason": "Prolonged hypoglycemia risk",
        },
        "canagliflozin": {
            "contraindicated_below": 30,
            "caution_below": 45,
            "reason": "Reduced efficacy",
        },
        "dapagliflozin": {
            "contraindicated_below": 25,
            "caution_below": 45,
            "reason": "Reduced efficacy",
        },
        "empagliflozin": {
            "contraindicated_below": 30,
            "caution_below": 45,
            "reason": "Reduced efficacy",
        },
        "sitagliptin": {
            "dose_adjust_below": 45,
            "caution_below": 45,
            "reason": "Dose reduction required",
        },
        "alogliptin": {
            "dose_adjust_below": 60,
            "caution_below": 60,
            "reason": "Dose reduction required",
        },
        "linagliptin": {"no_adjustment": True, "reason": "No renal adjustment needed"},
        "exenatide": {
            "contraindicated_below": 30,
            "reason": "Risk of acute kidney injury",
        },
        "gabapentin": {
            "dose_adjust_below": 60,
            "caution_below": 60,
            "reason": "Dose reduction required",
        },
        "pregabalin": {
            "dose_adjust_below": 60,
            "caution_below": 60,
            "reason": "Dose reduction required",
        },
        # Cardiovascular drugs requiring renal adjustment
        "verapamil": {
            "caution_below": 30,
            "reason": "Accumulation risk, dose reduction needed",
        },
        "diltiazem": {
            "caution_below": 30,
            "reason": "Accumulation risk, dose reduction needed",
        },
        "digoxin": {
            "contraindicated_below": 30,
            "caution_below": 60,
            "reason": "Toxicity risk - dose reduction required",
        },
        "sotalol": {"contraindicated_below": 40, "reason": "QT prolongation risk"},
        "atenolol": {
            "caution_below": 35,
            "reason": "Accumulation - dose reduction required",
        },
        "bisoprolol": {
            "caution_below": 20,
            "reason": "Dose reduction in severe renal impairment",
        },
        "acebutolol": {"caution_below": 50, "reason": "Dose reduction required"},
        "nadolol": {"caution_below": 50, "reason": "Dose reduction required"},
        # ACE inhibitors / ARBs - accumulation risk
        "lisinopril": {
            "caution_below": 30,
            "reason": "Start low dose, accumulation risk",
        },
        "enalapril": {
            "caution_below": 30,
            "reason": "Start low dose, accumulation risk",
        },
        "ramipril": {
            "caution_below": 40,
            "reason": "Start low dose, accumulation risk",
        },
        "captopril": {"caution_below": 30, "reason": "Dose reduction required"},
        # Diuretics
        "spironolactone": {
            "contraindicated_below": 30,
            "caution_below": 50,
            "reason": "Hyperkalemia risk",
        },
        "eplerenone": {"contraindicated_below": 30, "reason": "Hyperkalemia risk"},
        "furosemide": {
            "caution_below": 30,
            "reason": "May need higher doses for effect",
        },
        # Pain medications
        "morphine": {"caution_below": 30, "reason": "Active metabolites accumulate"},
        "codeine": {
            "caution_below": 30,
            "reason": "Metabolite accumulation, reduce dose",
        },
        "tramadol": {"caution_below": 30, "reason": "Reduce dose and frequency"},
        "oxycodone": {"caution_below": 30, "reason": "Dose reduction recommended"},
        # Antibiotics
        "ciprofloxacin": {"caution_below": 30, "reason": "Dose reduction required"},
        "levofloxacin": {"caution_below": 50, "reason": "Dose reduction required"},
        "nitrofurantoin": {
            "contraindicated_below": 30,
            "reason": "Ineffective and toxic",
        },
        "gentamicin": {
            "caution_below": 60,
            "reason": "Nephrotoxic, dose adjustment required",
        },
        "vancomycin": {"caution_below": 50, "reason": "Nephrotoxic, monitor levels"},
        # Anticoagulants
        "dabigatran": {
            "contraindicated_below": 30,
            "caution_below": 50,
            "reason": "Bleeding risk - dose reduction or avoid",
        },
        "rivaroxaban": {
            "contraindicated_below": 15,
            "caution_below": 50,
            "reason": "Dose reduction required",
        },
        "apixaban": {"caution_below": 25, "reason": "Dose reduction required"},
        "enoxaparin": {
            "caution_below": 30,
            "reason": "Bleeding risk - dose reduction required",
        },
        # Other common drugs
        "methotrexate": {
            "contraindicated_below": 30,
            "caution_below": 60,
            "reason": "Severe toxicity risk",
        },
        "allopurinol": {
            "caution_below": 60,
            "reason": "Start low dose, titrate slowly",
        },
        "colchicine": {
            "contraindicated_below": 30,
            "caution_below": 60,
            "reason": "Toxicity risk",
        },
        "lithium": {
            "contraindicated_below": 30,
            "caution_below": 60,
            "reason": "Toxicity - careful monitoring",
        },
        "baclofen": {"caution_below": 30, "reason": "Neurotoxicity risk"},
        "ranitidine": {"caution_below": 50, "reason": "Dose reduction required"},
        "famotidine": {"caution_below": 50, "reason": "Dose reduction required"},
    }

    # Drugs that need extra caution in ANY patient with eGFR < 30
    SEVERE_CKD_CAUTION_DRUGS = [
        "nsaids",
        "ibuprofen",
        "naproxen",
        "diclofenac",
        "celecoxib",
        "meloxicam",
        "aspirin",  # high dose
        "magnesium",
        "phosphate",
        "potassium",
        "contrast",
        "gadolinium",
    ]

    # ==================== DRUG CLASS PATTERNS ====================
    # These patterns match drug names by suffix/prefix to catch entire classes

    # Pattern: (suffix/contains, risk_action, reason, egfr_threshold)
    # risk_action: "contraindicated", "high_risk", "caution", "dose_adjust"
    DRUG_CLASS_PATTERNS = {
        # ACE Inhibitors (-pril)
        "pril": {
            "action": "caution",
            "egfr_below": 30,
            "reason": "ACE-I accumulation, hyperkalemia risk",
            "monitoring": ["potassium", "creatinine"],
        },
        # ARBs (-sartan)
        "sartan": {
            "action": "caution",
            "egfr_below": 30,
            "reason": "ARB accumulation, hyperkalemia risk",
            "monitoring": ["potassium", "creatinine"],
        },
        # Beta blockers (-olol)
        "olol": {
            "action": "caution",
            "egfr_below": 30,
            "reason": "Beta-blocker may accumulate, masks hypoglycemia",
            "monitoring": ["heart rate", "blood glucose"],
        },
        # Calcium channel blockers (-dipine, -pamil, -zem)
        "dipine": {
            "action": "caution",
            "egfr_below": 30,
            "reason": "CCB dose adjustment may be needed",
            "monitoring": ["blood pressure"],
        },
        "pamil": {
            "action": "caution",
            "egfr_below": 30,
            "reason": "Verapamil-type CCB accumulation risk",
            "monitoring": ["heart rate", "blood pressure"],
        },
        "azem": {
            "action": "caution",
            "egfr_below": 30,
            "reason": "Diltiazem-type CCB needs monitoring",
            "monitoring": ["heart rate"],
        },
        # Statins (-statin)
        "statin": {
            "action": "caution",
            "egfr_below": 30,
            "reason": "Some statins need dose adjustment in CKD",
            "monitoring": ["liver function", "muscle symptoms"],
        },
        # Fluoroquinolones (-floxacin)
        "floxacin": {
            "action": "dose_adjust",
            "egfr_below": 50,
            "reason": "Fluoroquinolone dose reduction required",
            "monitoring": ["tendon pain", "QT interval"],
        },
        # Aminoglycosides (-mycin, -micin)
        "mycin": {
            "action": "high_risk",
            "egfr_below": 60,
            "reason": "Aminoglycoside nephrotoxicity",
            "monitoring": ["drug levels", "creatinine", "hearing"],
        },
        "micin": {
            "action": "high_risk",
            "egfr_below": 60,
            "reason": "Aminoglycoside nephrotoxicity",
            "monitoring": ["drug levels", "creatinine"],
        },
        # Cephalosporins (cef-, -cef)
        "cef": {
            "action": "dose_adjust",
            "egfr_below": 50,
            "reason": "Cephalosporin dose adjustment needed",
            "monitoring": ["creatinine"],
        },
        # Penicillins (-cillin)
        "cillin": {
            "action": "dose_adjust",
            "egfr_below": 30,
            "reason": "Penicillin dose adjustment in severe CKD",
            "monitoring": ["creatinine"],
        },
        # Sulfonylureas (-ide ending for glyburide, glipizide, etc.)
        "glipizide": {
            "action": "caution",
            "egfr_below": 50,
            "reason": "Hypoglycemia risk with renal impairment",
            "monitoring": ["blood glucose"],
        },
        "gliclazide": {
            "action": "caution",
            "egfr_below": 40,
            "reason": "Hypoglycemia risk",
            "monitoring": ["blood glucose"],
        },
        "glimepiride": {
            "action": "caution",
            "egfr_below": 50,
            "reason": "Hypoglycemia risk",
            "monitoring": ["blood glucose"],
        },
        # SGLT2 inhibitors (-gliflozin)
        "gliflozin": {
            "action": "caution",
            "egfr_below": 45,
            "reason": "SGLT2i reduced efficacy in CKD",
            "monitoring": ["eGFR", "ketones"],
        },
        # DPP-4 inhibitors (-gliptin)
        "gliptin": {
            "action": "dose_adjust",
            "egfr_below": 45,
            "reason": "DPP-4i dose adjustment (except linagliptin)",
            "monitoring": ["blood glucose"],
        },
        # GLP-1 agonists (-glutide, -natide)
        "glutide": {
            "action": "caution",
            "egfr_below": 30,
            "reason": "GLP-1 RA caution in severe CKD",
            "monitoring": ["GI symptoms", "kidney function"],
        },
        "natide": {
            "action": "caution",
            "egfr_below": 30,
            "reason": "GLP-1 RA limited experience in severe CKD",
            "monitoring": ["kidney function"],
        },
        # Thiazide diuretics (-thiazide)
        "thiazide": {
            "action": "caution",
            "egfr_below": 30,
            "reason": "Thiazides less effective in severe CKD",
            "monitoring": ["electrolytes", "blood pressure"],
        },
        # Loop diuretics (-semide, -tanide)
        "semide": {
            "action": "dose_adjust",
            "egfr_below": 30,
            "reason": "Loop diuretic - may need higher doses",
            "monitoring": ["electrolytes", "volume status"],
        },
        # Potassium-sparing diuretics
        "spironolactone": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "Severe hyperkalemia risk",
            "monitoring": ["potassium"],
        },
        "eplerenone": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "Severe hyperkalemia risk",
            "monitoring": ["potassium"],
        },
        "amiloride": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "Severe hyperkalemia risk",
            "monitoring": ["potassium"],
        },
        "triamterene": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "Severe hyperkalemia risk",
            "monitoring": ["potassium"],
        },
        # Opioids (-done, -codone, -morphone)
        "codone": {
            "action": "dose_adjust",
            "egfr_below": 30,
            "reason": "Opioid metabolite accumulation",
            "monitoring": ["sedation", "respiratory rate"],
        },
        "morphone": {
            "action": "dose_adjust",
            "egfr_below": 30,
            "reason": "Opioid metabolite accumulation",
            "monitoring": ["sedation", "respiratory rate"],
        },
        "done": {
            "action": "caution",
            "egfr_below": 30,
            "reason": "Opioid caution in renal impairment",
            "monitoring": ["sedation"],
        },
        # Gabapentinoids
        "gabapentin": {
            "action": "dose_adjust",
            "egfr_below": 60,
            "reason": "Gabapentin dose reduction required",
            "monitoring": ["sedation", "dizziness"],
        },
        "pregabalin": {
            "action": "dose_adjust",
            "egfr_below": 60,
            "reason": "Pregabalin dose reduction required",
            "monitoring": ["sedation", "dizziness"],
        },
        # Direct oral anticoagulants (-xaban, -gatran)
        "xaban": {
            "action": "dose_adjust",
            "egfr_below": 50,
            "reason": "DOAC dose reduction, bleeding risk",
            "monitoring": ["bleeding", "creatinine"],
        },
        "gatran": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "Dabigatran contraindicated in severe CKD",
            "monitoring": ["bleeding"],
        },
        # Antivirals (-vir)
        "ciclovir": {
            "action": "dose_adjust",
            "egfr_below": 50,
            "reason": "Acyclovir/valacyclovir dose reduction",
            "monitoring": ["neurological symptoms"],
        },
        "vir": {
            "action": "caution",
            "egfr_below": 50,
            "reason": "Antiviral may need dose adjustment",
            "monitoring": ["kidney function"],
        },
        # Antifungals (-azole)
        "azole": {
            "action": "caution",
            "egfr_below": 50,
            "reason": "Azole antifungal - check specific agent",
            "monitoring": ["liver function", "drug interactions"],
        },
        # Proton pump inhibitors (-prazole)
        "prazole": {
            "action": "caution",
            "egfr_below": 30,
            "reason": "PPI long-term use associated with CKD progression",
            "monitoring": ["magnesium", "B12"],
        },
        # H2 blockers (-tidine)
        "tidine": {
            "action": "dose_adjust",
            "egfr_below": 50,
            "reason": "H2 blocker dose reduction needed",
            "monitoring": ["confusion in elderly"],
        },
        # Bisphosphonates (-dronate)
        "dronate": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "Bisphosphonate contraindicated in severe CKD",
            "monitoring": ["calcium", "vitamin D"],
        },
        # NSAIDs (common names)
        "ibuprofen": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "NSAID nephrotoxicity",
            "monitoring": [],
        },
        "naproxen": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "NSAID nephrotoxicity",
            "monitoring": [],
        },
        "diclofenac": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "NSAID nephrotoxicity",
            "monitoring": [],
        },
        "celecoxib": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "NSAID nephrotoxicity",
            "monitoring": [],
        },
        "meloxicam": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "NSAID nephrotoxicity",
            "monitoring": [],
        },
        "ketorolac": {
            "action": "contraindicated",
            "egfr_below": 60,
            "reason": "NSAID - high nephrotoxicity risk",
            "monitoring": [],
        },
        "indomethacin": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "NSAID nephrotoxicity",
            "monitoring": [],
        },
        "piroxicam": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "NSAID nephrotoxicity",
            "monitoring": [],
        },
        # Metformin
        "metformin": {
            "action": "contraindicated",
            "egfr_below": 30,
            "reason": "Lactic acidosis risk",
            "monitoring": [],
        },
    }

    # Potassium-raising drugs (caution with ACE-I/ARB in diabetics)
    HYPERKALEMIA_RISK_DRUGS = [
        "lisinopril",
        "enalapril",
        "ramipril",
        "captopril",  # ACE inhibitors
        "losartan",
        "valsartan",
        "irbesartan",
        "candesartan",  # ARBs
        "spironolactone",
        "eplerenone",  # Aldosterone antagonists
        "trimethoprim",
        "potassium supplements",
    ]

    # Hepatotoxic drugs (caution with fatty liver disease common in T2DM)
    HEPATOTOXIC_DRUGS = [
        "acetaminophen",
        "amiodarone",
        "methotrexate",
        "isoniazid",
        "valproic acid",
        "phenytoin",
        "ketoconazole",
        "statins",
    ]

    # Drugs beneficial for diabetics (prefer these)
    CARDIOPROTECTIVE_IN_DIABETES = [
        "empagliflozin",
        "canagliflozin",
        "dapagliflozin",  # SGLT2i
        "liraglutide",
        "semaglutide",
        "dulaglutide",  # GLP-1 RA
        "metformin",  # First-line
        "lisinopril",
        "enalapril",
        "ramipril",  # ACE-I for nephroprotection
        "atorvastatin",
        "rosuvastatin",  # Statins
    ]

    # Weight gain drugs to avoid in obese diabetics
    WEIGHT_GAIN_DRUGS = [
        "insulin",
        "sulfonylureas",
        "glimepiride",
        "glipizide",
        "glyburide",
        "thiazolidinediones",
        "pioglitazone",
        "rosiglitazone",
        "mirtazapine",
        "olanzapine",
        "clozapine",
        "prednisone",
        "dexamethasone",
    ]

    # Fatal combinations in diabetics
    FATAL_COMBINATIONS = [
        {
            "drugs": ["metformin", "iodinated contrast"],
            "condition": "egfr < 30",
            "reason": "High risk of contrast-induced nephropathy and lactic acidosis",
        },
        {
            "drugs": ["potassium supplements", "spironolactone", "ace_inhibitor"],
            "condition": "egfr < 30",
            "reason": "Severe hyperkalemia risk - potentially fatal",
        },
    ]

    EVIDENCE_TAGS = {
        "renal": "Label/renal guidance",
        "hypoglycemia": "Class hypoglycemia risk",
        "hyperglycemia": "Class hyperglycemia risk",
        "mask_hypo": "Beta-blocker masking",
        "hyperkalemia": "K+ elevation risk",
        "nephrotoxic": "Nephrotoxic agent",
        "hepatotoxic": "Hepatotoxic agent",
        "weight_gain": "Weight gain risk",
        "hf_tzd": "HF caution (TZD)",
        "glp1_pancreatitis": "GLP-1 GI/pancreatitis caution",
        "alcohol_hypo": "Alcohol-induced hypoglycemia",
    }

    def __init__(self):
        """Initialize rules engine and load data from JSON file."""
        self.rules_loaded = True
        self.json_data = self._load_json_data()
        self._integrate_json_data()

    def _load_json_data(self) -> Dict:
        """Load diabetes medications data from JSON file."""
        try:
            if _MEDICATIONS_JSON_PATH.exists():
                with open(_MEDICATIONS_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(
                    f"Loaded diabetes medications data from {_MEDICATIONS_JSON_PATH}"
                )
                return data
            else:
                logger.warning(
                    f"JSON file not found at {_MEDICATIONS_JSON_PATH}, using hardcoded rules only"
                )
                return {}
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}, using hardcoded rules only")
            return {}

    def _integrate_json_data(self):
        """Integrate JSON data into the rules engine."""
        if not self.json_data:
            return

        # 1. Build hypoglycemia risk drugs from diabetes_medications
        if "diabetes_medications" in self.json_data:
            for class_name, class_data in self.json_data[
                "diabetes_medications"
            ].items():
                drugs = class_data.get("drugs", [])
                hypoglycemia_risk = class_data.get("hypoglycemia_risk", "low")

                if hypoglycemia_risk in ["high", "moderate"]:
                    # Add to hypoglycemia risk drugs
                    if class_name not in self.HYPOGLYCEMIA_RISK_DRUGS:
                        self.HYPOGLYCEMIA_RISK_DRUGS[class_name] = []
                    for drug in drugs:
                        if drug.lower() not in [
                            d.lower() for d in self.HYPOGLYCEMIA_RISK_DRUGS[class_name]
                        ]:
                            self.HYPOGLYCEMIA_RISK_DRUGS[class_name].append(drug)

        # 2. Build hyperglycemia risk drugs from drugs_affecting_glucose
        if "drugs_affecting_glucose" in self.json_data:
            increase_glucose = self.json_data["drugs_affecting_glucose"].get(
                "increase_glucose", []
            )
            for item in increase_glucose:
                drug = item.get("drug", "").lower()
                drug_class = item.get("class", "")
                if drug and drug_class:
                    if drug_class not in self.HYPERGLYCEMIA_RISK_DRUGS:
                        self.HYPERGLYCEMIA_RISK_DRUGS[drug_class] = []
                    if drug not in [
                        d.lower() for d in self.HYPERGLYCEMIA_RISK_DRUGS[drug_class]
                    ]:
                        self.HYPERGLYCEMIA_RISK_DRUGS[drug_class].append(
                            item.get("drug")
                        )

        # 3. Integrate eGFR dosing guidance
        if "egfr_dosing_guidance" in self.json_data:
            for drug_name, guidance in self.json_data["egfr_dosing_guidance"].items():
                drug_lower = drug_name.lower()
                # Parse guidance and add to EGFR_CONTRAINDICATIONS if not already present
                if drug_lower not in self.EGFR_CONTRAINDICATIONS:
                    # Try to infer thresholds from guidance text
                    if "Contraindicated" in guidance.get("below_30", ""):
                        self.EGFR_CONTRAINDICATIONS[drug_lower] = {
                            "contraindicated_below": 30,
                            "reason": guidance.get(
                                "below_30", "Renal adjustment required"
                            ),
                        }
                    elif "30_to_45" in guidance or "30_to_60" in guidance:
                        caution_threshold = 45 if "30_to_45" in guidance else 60
                        self.EGFR_CONTRAINDICATIONS[drug_lower] = {
                            "caution_below": caution_threshold,
                            "reason": f"Renal dosing: {guidance.get('30_to_45') or guidance.get('30_to_60', 'Dose adjustment needed')}",
                        }

        # 4. Integrate dangerous combinations
        if "dangerous_combinations" in self.json_data:
            for combo in self.json_data["dangerous_combinations"]:
                drugs = combo.get("drugs", [])
                risk = combo.get("risk", "")
                guidance = combo.get("guidance", "")

                # Add to FATAL_COMBINATIONS if not already present
                combo_normalized = sorted([d.lower() for d in drugs])
                existing_combo = False
                for existing in self.FATAL_COMBINATIONS:
                    existing_drugs = sorted(
                        [d.lower() for d in existing.get("drugs", [])]
                    )
                    if combo_normalized == existing_drugs:
                        existing_combo = True
                        break

                if not existing_combo:
                    self.FATAL_COMBINATIONS.append(
                        {
                            "drugs": drugs,
                            "condition": combo.get("condition", ""),
                            "reason": f"{risk}: {guidance}",
                        }
                    )

        # 5. Build additional drug lists from common_comorbidity_medications
        if "common_comorbidity_medications" in self.json_data:
            # Add ACE inhibitors and ARBs to hyperkalemia risk if not already there
            for class_name, class_data in self.json_data[
                "common_comorbidity_medications"
            ].items():
                drugs = class_data.get("drugs", [])
                if class_name in ["ace_inhibitors", "arbs"]:
                    for drug in drugs:
                        if drug.lower() not in [
                            d.lower() for d in self.HYPERKALEMIA_RISK_DRUGS
                        ]:
                            self.HYPERKALEMIA_RISK_DRUGS.append(drug)

                # Add beta blockers to mask hypoglycemia list
                if class_name == "beta_blockers":
                    for drug in drugs:
                        if drug.lower() not in [
                            d.lower() for d in self.MASK_HYPOGLYCEMIA
                        ]:
                            self.MASK_HYPOGLYCEMIA.append(drug)

                # Add cardioprotective drugs
                if class_name in ["ace_inhibitors", "statins"]:
                    for drug in drugs:
                        if drug.lower() not in [
                            d.lower() for d in self.CARDIOPROTECTIVE_IN_DIABETES
                        ]:
                            self.CARDIOPROTECTIVE_IN_DIABETES.append(drug)

        logger.info("JSON data integrated into rules engine")

    def assess_drug_risk(
        self,
        drug_name: str,
        patient: Dict,
        current_medications: Optional[List[str]] = None,
    ) -> RiskAssessment:
        """
        Assess the risk of a drug for a specific diabetic patient.

        Args:
            drug_name: Name of the drug to assess
            patient: Patient data dict with labs, complications, etc.
            current_medications: List of current medication names

        Returns:
            RiskAssessment with risk level, factors, and recommendations
        """
        drug_lower = drug_name.lower().strip()
        current_meds = [m.lower().strip() for m in (current_medications or [])]

        risk_factors = []
        rule_refs = []
        evidence_sources = []
        patient_factors = []
        risk_score = 0
        monitoring = []
        alternatives = []
        interactions = []

        # 1. Check eGFR-based contraindications
        egfr = patient.get("egfr")
        specific_rule_applied = False

        # 1a. Check specific drug-eGFR rules
        if egfr and drug_lower in self.EGFR_CONTRAINDICATIONS:
            specific_rule_applied = True
            rule = self.EGFR_CONTRAINDICATIONS[drug_lower]
            if egfr < rule.get("contraindicated_below", 0):
                # Get dosing guidance from JSON if available
                dosing_guidance = self._get_egfr_dosing_guidance(drug_lower, egfr)
                recommendation_text = f"CONTRAINDICATED: {rule['reason']}. Do not use."
                if dosing_guidance:
                    recommendation_text += f" {dosing_guidance}"

                return RiskAssessment(
                    drug_name=drug_name,
                    risk_level="contraindicated",
                    severity="contraindicated",
                    risk_score=100,
                    risk_factors=[
                        f"eGFR {egfr} < {rule['contraindicated_below']}: {rule['reason']}"
                    ],
                    rule_references=[
                        f"Renal threshold: {drug_lower} contraindicated if eGFR < {rule['contraindicated_below']}"
                    ],
                    evidence_sources=[self.EVIDENCE_TAGS["renal"]],
                    patient_factors=[f"eGFR={egfr}"],
                    recommendation=recommendation_text,
                    alternatives=self._get_alternatives(drug_lower, patient),
                    monitoring=[],
                    interactions=[],
                )
            elif egfr < rule.get("caution_below", 0):
                dosing_guidance = self._get_egfr_dosing_guidance(drug_lower, egfr)
                risk_factors.append(f"eGFR {egfr} requires caution: {rule['reason']}")
                if dosing_guidance:
                    risk_factors.append(f"Dosing guidance: {dosing_guidance}")
                rule_refs.append(
                    f"Renal adjustment: {drug_lower} caution if eGFR < {rule['caution_below']}"
                )
                evidence_sources.append(self.EVIDENCE_TAGS["renal"])
                patient_factors.append(f"eGFR={egfr}")
                risk_score += 40
                monitoring.append("Monitor kidney function closely")
            elif egfr < rule.get("dose_adjust_below", 0):
                dosing_guidance = self._get_egfr_dosing_guidance(drug_lower, egfr)
                risk_factors.append(
                    f"eGFR {egfr} - dose adjustment needed: {rule['reason']}"
                )
                if dosing_guidance:
                    risk_factors.append(f"Dosing guidance: {dosing_guidance}")
                rule_refs.append(f"Renal dose adjustment for {drug_lower}")
                evidence_sources.append(self.EVIDENCE_TAGS["renal"])
                patient_factors.append(f"eGFR={egfr}")
                risk_score += 25
                monitoring.append("Monitor kidney function, adjust dose")

        # 1a-json. Check eGFR dosing guidance from JSON (if not in hardcoded rules)
        elif egfr and self.json_data and "egfr_dosing_guidance" in self.json_data:
            dosing_guidance = self._get_egfr_dosing_guidance(drug_lower, egfr)
            if dosing_guidance:
                risk_factors.append(f"eGFR {egfr} - {dosing_guidance}")
                rule_refs.append(f"Renal dosing guidance from clinical data")
                evidence_sources.append(self.EVIDENCE_TAGS["renal"])
                patient_factors.append(f"eGFR={egfr}")
                risk_score += 20
                monitoring.append("Monitor kidney function and adjust dose as needed")

        # 1b. Check severe CKD caution drugs (eGFR < 30)
        if egfr and egfr < 30:
            for caution_drug in self.SEVERE_CKD_CAUTION_DRUGS:
                if caution_drug in drug_lower or drug_lower in caution_drug:
                    risk_factors.append(
                        f"Severe kidney disease (eGFR {egfr}) - {drug_name} requires extreme caution"
                    )
                    rule_refs.append("Severe CKD caution")
                    evidence_sources.append(self.EVIDENCE_TAGS["renal"])
                    patient_factors.append(f"eGFR={egfr} (Stage 4/5 CKD)")
                    risk_score += 50
                    monitoring.append("Close renal function monitoring")
                    break

        # 1c. Check drug class patterns for renal dosing
        pattern_matched = False
        # Only check class patterns if we haven't already applied a specific rule
        if not specific_rule_applied:
            for pattern, rule in self.DRUG_CLASS_PATTERNS.items():
                if pattern in drug_lower:
                    if egfr and egfr < rule.get("egfr_below", 0):
                        pattern_matched = True
                    action = rule["action"]
                    reason = rule["reason"]

                    if action == "contraindicated":
                        return RiskAssessment(
                            drug_name=drug_name,
                            risk_level="contraindicated",
                            severity="contraindicated",
                            risk_score=100,
                            risk_factors=[
                                f"eGFR {egfr} < {rule['egfr_below']}: {reason}"
                            ],
                            rule_references=[
                                f"Drug class pattern: {pattern} contraindicated if eGFR < {rule['egfr_below']}"
                            ],
                            evidence_sources=[self.EVIDENCE_TAGS["renal"]],
                            patient_factors=[f"eGFR={egfr}"],
                            recommendation=f"CONTRAINDICATED: {reason}. Do not use.",
                            alternatives=self._get_alternatives(drug_lower, patient),
                            monitoring=[],
                            interactions=[],
                        )
                    elif action == "high_risk":
                        risk_factors.append(f"Drug class ({pattern}): {reason}")
                        rule_refs.append(f"Drug class renal rule: {pattern}")
                        evidence_sources.append(self.EVIDENCE_TAGS["renal"])
                        patient_factors.append(f"eGFR={egfr}")
                        risk_score += 45
                        monitoring.extend(rule.get("monitoring", []))
                    elif action == "caution":
                        risk_factors.append(f"Drug class ({pattern}): {reason}")
                        rule_refs.append(f"Drug class renal rule: {pattern}")
                        evidence_sources.append(self.EVIDENCE_TAGS["renal"])
                        patient_factors.append(f"eGFR={egfr}")
                        risk_score += 35
                        monitoring.extend(rule.get("monitoring", []))
                    elif action == "dose_adjust":
                        risk_factors.append(f"Drug class ({pattern}): {reason}")
                        rule_refs.append(f"Drug class dose adjustment: {pattern}")
                        evidence_sources.append(self.EVIDENCE_TAGS["renal"])
                        patient_factors.append(f"eGFR={egfr}")
                        risk_score += 25
                        monitoring.extend(rule.get("monitoring", []))
                        break  # Stop after first pattern match

        # 1d. General severe kidney function warning (eGFR < 30) for unmatched drugs
        if (
            egfr
            and egfr < 30
            and not pattern_matched
            and drug_lower not in self.EGFR_CONTRAINDICATIONS
        ):
            # Unknown drug in severe CKD - flag for review with higher score
            risk_factors.append(
                f"WARNING: Severe CKD (eGFR {egfr}): Drug not in renal database - REVIEW REQUIRED"
            )
            rule_refs.append("Severe CKD - unknown renal handling")
            evidence_sources.append(self.EVIDENCE_TAGS["renal"])
            patient_factors.append(f"eGFR={egfr} (Stage 4/5 CKD)")
            risk_score += 40  # Higher score for unknown drugs in severe CKD
            monitoring.append("Pharmacist consult recommended for renal dosing")
            monitoring.append("Monitor kidney function closely")

        # 1e. Moderate CKD warning (eGFR 30-59) for unmatched drugs
        elif (
            egfr
            and 30 <= egfr < 60
            and not pattern_matched
            and drug_lower not in self.EGFR_CONTRAINDICATIONS
        ):
            risk_factors.append(
                f"Moderate CKD (eGFR {egfr}): Consider renal dosing review"
            )
            rule_refs.append("Moderate CKD - consider dose adjustment")
            evidence_sources.append(self.EVIDENCE_TAGS["renal"])
            patient_factors.append(f"eGFR={egfr} (Stage 3 CKD)")
            risk_score += 15
            monitoring.append("Monitor kidney function")

        # 2. Check hypoglycemia risk
        if self._is_hypoglycemia_risk_drug(drug_lower):
            risk_factors.append("Drug increases hypoglycemia risk")
            rule_refs.append("Hypoglycemia-prone class")
            evidence_sources.append(self.EVIDENCE_TAGS["hypoglycemia"])
            risk_score += 25
            monitoring.append("Monitor blood glucose frequently")

            # Higher risk if on other hypoglycemia drugs
            for med in current_meds:
                if self._is_hypoglycemia_risk_drug(med):
                    risk_factors.append(
                        f"Combined with {med} - increased hypoglycemia risk"
                    )
                    rule_refs.append(f"Combination hypoglycemia risk with {med}")
                    evidence_sources.append(self.EVIDENCE_TAGS["hypoglycemia"])
                    risk_score += 20

        # 3. Check if drug masks hypoglycemia
        if drug_lower in self.MASK_HYPOGLYCEMIA:
            risk_factors.append("May mask hypoglycemia symptoms (tachycardia, tremor)")
            rule_refs.append("Beta blocker masking hypoglycemia symptoms")
            evidence_sources.append(self.EVIDENCE_TAGS["mask_hypo"])
            risk_score += 30
            monitoring.append("Educate patient on atypical hypoglycemia symptoms")
            alternatives.extend(["Cardioselective beta-blockers may be safer"])

        # 4. Check hyperglycemia risk
        if self._is_hyperglycemia_risk_drug(drug_lower):
            risk_factors.append("Drug may worsen blood glucose control")
            rule_refs.append("Hyperglycemia-worsening class")
            evidence_sources.append(self.EVIDENCE_TAGS["hyperglycemia"])
            risk_score += 20
            monitoring.append("Monitor HbA1c and fasting glucose")

        # 5. Check nephrotoxicity with diabetic nephropathy
        if patient.get("has_nephropathy") and self._is_nephrotoxic(drug_lower):
            risk_factors.append("Nephrotoxic drug in patient with diabetic nephropathy")
            rule_refs.append("Nephrotoxic agent with nephropathy")
            evidence_sources.append(self.EVIDENCE_TAGS["nephrotoxic"])
            patient_factors.append("Nephropathy")
            risk_score += 40
            monitoring.append("Monitor creatinine and eGFR closely")

        # 6. Check hyperkalemia risk
        potassium = patient.get("potassium")
        if drug_lower in [d.lower() for d in self.HYPERKALEMIA_RISK_DRUGS]:
            if potassium and potassium > 5.0:
                risk_factors.append(f"Hyperkalemia risk - current K+ is {potassium}")
                rule_refs.append("Hyperkalemia risk agent with elevated K+")
                evidence_sources.append(self.EVIDENCE_TAGS["hyperkalemia"])
                patient_factors.append(f"K+={potassium}")
                risk_score += 35
            elif potassium and potassium > 4.5:
                risk_factors.append("Monitor potassium - already borderline elevated")
                rule_refs.append("Hyperkalemia risk agent with borderline K+")
                evidence_sources.append(self.EVIDENCE_TAGS["hyperkalemia"])
                patient_factors.append(f"K+={potassium}")
                risk_score += 15
            monitoring.append("Monitor serum potassium")

            # Check for dangerous combinations
            for med in current_meds:
                if med in [d.lower() for d in self.HYPERKALEMIA_RISK_DRUGS]:
                    risk_factors.append(f"Combined with {med} - high hyperkalemia risk")
                    rule_refs.append(f"Combination hyperkalemia risk with {med}")
                    evidence_sources.append(self.EVIDENCE_TAGS["hyperkalemia"])
                    risk_score += 25

        # 7. Check cardiovascular considerations
        if patient.get("has_cardiovascular"):
            if drug_lower in [d.lower() for d in self.CARDIOPROTECTIVE_IN_DIABETES]:
                risk_factors.append("Cardioprotective - beneficial for this patient")
                risk_score -= 10  # Bonus for beneficial drug

            # TZDs contraindicated in heart failure
            if drug_lower in ["pioglitazone", "rosiglitazone"]:
                risk_factors.append("Thiazolidinediones may worsen heart failure")
                rule_refs.append("HF caution with TZD")
                evidence_sources.append(self.EVIDENCE_TAGS["hf_tzd"])
                patient_factors.append("Heart failure")
                risk_score += 50

        # 8. Check weight considerations
        if patient.get("has_obesity"):
            if drug_lower in [d.lower() for d in self.WEIGHT_GAIN_DRUGS]:
                risk_factors.append(
                    "Drug may cause weight gain - consider alternatives"
                )
                rule_refs.append("Weight-gain risk in obesity")
                evidence_sources.append(self.EVIDENCE_TAGS["weight_gain"])
                patient_factors.append("Obesity")
                risk_score += 15
                alternatives.extend(["GLP-1 agonists", "SGLT2 inhibitors", "Metformin"])

        # 9. Check hepatotoxicity
        alt = patient.get("alt")
        ast = patient.get("ast")
        if alt and alt > 80 or ast and ast > 80:  # 2x upper limit
            if drug_lower in [d.lower() for d in self.HEPATOTOXIC_DRUGS]:
                risk_factors.append("Hepatotoxic drug with elevated liver enzymes")
                rule_refs.append("Hepatotoxic agent with elevated LFTs")
                evidence_sources.append(self.EVIDENCE_TAGS["hepatotoxic"])
                patient_factors.append(f"ALT/AST high")
                risk_score += 35
                monitoring.append("Monitor liver function tests")

        # 10. Check dangerous combinations from JSON
        dangerous_combo = self._check_dangerous_combinations(drug_lower, current_meds)
        if dangerous_combo:
            risk_factors.append(f"Dangerous combination: {dangerous_combo['reason']}")
            rule_refs.append(
                f"Dangerous combination: {', '.join(dangerous_combo['drugs'])}"
            )
            evidence_sources.append("Clinical guideline - dangerous combination")
            risk_score += 60
            interactions.append(
                {
                    "drugs": dangerous_combo["drugs"],
                    "risk": dangerous_combo.get("risk", "High risk"),
                    "guidance": dangerous_combo.get("guidance", ""),
                }
            )

        # 11. Alcohol + hypoglycemia risk
        if drug_lower == "alcohol":
            risk_factors.append(
                "Alcohol increases hypoglycemia risk with insulin/secretagogues"
            )
            rule_refs.append("Alcohol hypoglycemia risk")
            evidence_sources.append(self.EVIDENCE_TAGS["alcohol_hypo"])
            risk_score += 20

        # 12. GLP-1 GI/pancreatitis caution
        if drug_lower in [
            "liraglutide",
            "semaglutide",
            "dulaglutide",
            "exenatide",
            "tirzepatide",
        ]:
            risk_factors.append("GLP-1 RA: GI/pancreatitis caution; monitor symptoms")
            rule_refs.append("GLP-1 GI/pancreatitis caution")
            evidence_sources.append(self.EVIDENCE_TAGS["glp1_pancreatitis"])
            monitoring.append("Monitor abdominal pain, pancreatitis symptoms")

        # 13. Check drug class warnings from JSON
        # Only add generic class warnings if we aren't already at a critical risk level
        # to avoid double-counting risks that were likely already caught by specific rules
        if risk_score < 60:
            drug_class_info = self._get_drug_class_info(drug_lower)
            if drug_class_info:
                warnings = drug_class_info.get("warnings", [])
                if warnings:
                    for warning in warnings:
                        risk_factors.append(f"Class warning: {warning}")
                        # Do not add to risk_score for generic warnings to avoid over-alerting
                        # risk_score += 15
                        monitoring.append("Monitor for class-specific adverse effects")

        # 10. Calculate final risk level
        risk_level = self._score_to_risk_level(risk_score)
        severity = self._risk_level_to_severity(risk_level)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            drug_name, risk_level, risk_factors, monitoring
        )

        return RiskAssessment(
            drug_name=drug_name,
            risk_level=risk_level,
            severity=severity,
            risk_score=min(100, max(0, risk_score)),
            risk_factors=risk_factors,
            rule_references=list(
                dict.fromkeys(rule_refs)
            ),  # de-dup while preserving order
            evidence_sources=list(dict.fromkeys(evidence_sources)),
            patient_factors=list(dict.fromkeys(patient_factors)),
            recommendation=recommendation,
            alternatives=list(set(alternatives)),
            monitoring=list(set(monitoring)),
            interactions=interactions,
        )

    def check_medication_list(
        self, medications: List[str], patient: Dict
    ) -> List[RiskAssessment]:
        """Assess all medications for a patient."""
        results = []
        for med in medications:
            other_meds = [m for m in medications if m != med]
            assessment = self.assess_drug_risk(med, patient, other_meds)
            results.append(assessment)
        return results

    def find_safe_alternatives(
        self,
        drug_name: str,
        patient: Dict,
        current_medications: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Find safer alternatives for a drug."""
        alternatives = self._get_alternatives(drug_name.lower(), patient)

        safe_options = []
        for alt in alternatives:
            assessment = self.assess_drug_risk(alt, patient, current_medications)
            if assessment.risk_level in ["safe", "caution"]:
                safe_options.append(
                    {
                        "drug": alt,
                        "risk_level": assessment.risk_level,
                        "risk_score": assessment.risk_score,
                        "considerations": assessment.risk_factors,
                    }
                )

        return sorted(safe_options, key=lambda x: x["risk_score"])

    def _is_hypoglycemia_risk_drug(self, drug: str) -> bool:
        """Check if drug increases hypoglycemia risk."""
        for category, drugs in self.HYPOGLYCEMIA_RISK_DRUGS.items():
            if drug in [d.lower() for d in drugs]:
                return True
            if any(d.lower() in drug for d in drugs):
                return True
        return False

    def _is_hyperglycemia_risk_drug(self, drug: str) -> bool:
        """Check if drug worsens glucose control."""
        for category, drugs in self.HYPERGLYCEMIA_RISK_DRUGS.items():
            if drug in [d.lower() for d in drugs]:
                return True
        return False

    def _is_nephrotoxic(self, drug: str) -> bool:
        """Check if drug is nephrotoxic."""
        return drug in [d.lower() for d in self.NEPHROTOXIC_DRUGS]

    def _score_to_risk_level(self, score: float) -> str:
        """Convert numeric score to risk level."""
        if score >= 80:
            return "fatal"
        elif score >= 60:
            return "contraindicated"
        elif score >= 40:
            return "high_risk"
        elif score >= 20:
            return "caution"
        else:
            return "safe"

    def _risk_level_to_severity(self, risk_level: str) -> str:
        """Map risk level to standardized severity."""
        mapping = {
            "fatal": "fatal",
            "contraindicated": "contraindicated",
            "high_risk": "major",
            "caution": "moderate",
            "safe": "minor",
        }
        return mapping.get(risk_level, "unknown")

    def _generate_recommendation(
        self, drug: str, risk_level: str, factors: List[str], monitoring: List[str]
    ) -> str:
        """Generate recommendation text."""
        if risk_level == "fatal":
            return f"FATAL RISK: Do not use {drug}. " + "; ".join(factors[:2])
        elif risk_level == "contraindicated":
            return f"CONTRAINDICATED: Avoid {drug}. " + "; ".join(factors[:2])
        elif risk_level == "high_risk":
            return (
                f"HIGH RISK: Use {drug} only if no alternatives. "
                + f"Monitor: {', '.join(monitoring[:2])}"
            )
        elif risk_level == "caution":
            return (
                f"CAUTION: {drug} may be used with monitoring. "
                + f"Watch: {', '.join(monitoring[:2])}"
            )
        else:
            return f"SAFE: {drug} is generally safe for this patient."

    def _check_dangerous_combinations(
        self, drug: str, current_meds: List[str]
    ) -> Optional[Dict]:
        """Check if drug forms a dangerous combination with current medications."""
        all_meds = [drug] + current_meds
        all_meds_lower = [m.lower() for m in all_meds]

        for combo in self.FATAL_COMBINATIONS:
            combo_drugs = [d.lower() for d in combo.get("drugs", [])]
            # Check if all drugs in the combination are present
            if all(
                combo_drug in all_meds_lower
                or any(combo_drug in med for med in all_meds_lower)
                for combo_drug in combo_drugs
            ):
                return {
                    "drugs": combo.get("drugs", []),
                    "risk": combo.get("risk", "High risk"),
                    "reason": combo.get("reason", ""),
                    "guidance": combo.get("guidance", ""),
                }
        return None

    def _get_drug_class_info(self, drug: str) -> Optional[Dict]:
        """Get drug class information from JSON data."""
        if not self.json_data or "diabetes_medications" not in self.json_data:
            return None

        drug_lower = drug.lower()

        # Check diabetes medications
        for class_name, class_data in self.json_data["diabetes_medications"].items():
            drugs = [d.lower() for d in class_data.get("drugs", [])]
            if drug_lower in drugs or any(d in drug_lower for d in drugs):
                return class_data

        # Check comorbidity medications
        if "common_comorbidity_medications" in self.json_data:
            for class_name, class_data in self.json_data[
                "common_comorbidity_medications"
            ].items():
                drugs = [d.lower() for d in class_data.get("drugs", [])]
                if drug_lower in drugs or any(d in drug_lower for d in drugs):
                    return class_data

        return None

    def _get_egfr_dosing_guidance(self, drug: str, egfr: float) -> Optional[str]:
        """Get eGFR-based dosing guidance from JSON data."""
        if not self.json_data or "egfr_dosing_guidance" not in self.json_data:
            return None

        drug_lower = drug.lower()
        guidance_data = self.json_data["egfr_dosing_guidance"].get(drug_lower)

        if not guidance_data:
            return None

        # Return appropriate guidance based on eGFR range
        if egfr < 15:
            return guidance_data.get("below_15") or guidance_data.get("below_30")
        elif egfr < 30:
            return guidance_data.get("below_30")
        elif egfr < 45:
            return guidance_data.get("30_to_45") or guidance_data.get("30_to_60")
        elif egfr < 60:
            return guidance_data.get("30_to_60") or guidance_data.get("above_45")
        else:
            return guidance_data.get("above_60") or guidance_data.get("above_45")

    def _get_alternatives(self, drug: str, patient: Dict) -> List[str]:
        """Get safer alternatives for a drug class."""
        alternatives_map = {
            # Sulfonylureas -> safer options
            "glimepiride": ["sitagliptin", "linagliptin", "metformin"],
            "glipizide": ["sitagliptin", "linagliptin", "metformin"],
            "glyburide": ["glimepiride", "sitagliptin", "metformin"],
            # Beta-blockers -> cardioselective
            "propranolol": ["metoprolol succinate", "bisoprolol", "carvedilol"],
            "atenolol": ["metoprolol succinate", "bisoprolol"],
            "bisoprolol": ["metoprolol succinate", "nebivolol"],
            # NSAIDs -> safer pain options
            "ibuprofen": ["acetaminophen", "topical diclofenac"],
            "naproxen": ["acetaminophen", "topical diclofenac"],
            # TZDs -> safer options
            "pioglitazone": ["metformin", "empagliflozin", "liraglutide"],
            "rosiglitazone": ["metformin", "empagliflozin", "liraglutide"],
        }

        # Try to get alternatives from JSON data
        drug_class_info = self._get_drug_class_info(drug)
        if drug_class_info and self.json_data:
            # Look for safer alternatives in other drug classes
            alternatives = []

            # If it's a weight-gain drug, suggest weight-loss alternatives
            if drug_class_info.get("weight_effect") == "gain":
                for class_name, class_data in self.json_data.get(
                    "diabetes_medications", {}
                ).items():
                    if class_data.get("weight_effect") == "loss":
                        alternatives.extend(class_data.get("drugs", []))

            # If it has high hypoglycemia risk, suggest low-risk alternatives
            if drug_class_info.get("hypoglycemia_risk") == "high":
                for class_name, class_data in self.json_data.get(
                    "diabetes_medications", {}
                ).items():
                    if class_data.get("hypoglycemia_risk") == "low":
                        alternatives.extend(class_data.get("drugs", []))

            if alternatives:
                return alternatives[:5]  # Limit to top 5

        return alternatives_map.get(drug, [])
