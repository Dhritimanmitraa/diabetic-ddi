"""
Drug Name Validator.

Validates if a drug name is a known/recognized medication before processing.
This prevents the LLM from analyzing nonsense inputs like "lol".
"""

import json
import re
from pathlib import Path
from typing import Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)

# Load the medications JSON
_DATA_DIR = Path(__file__).parent / "data"
_MEDICATIONS_JSON_PATH = _DATA_DIR / "diabetes_medications.json"

# Known drug name patterns (suffixes that indicate real drugs)
DRUG_SUFFIXES = {
    # Diabetes medications
    "formin", "gliptin", "gliflozin", "glutide", "natide", "glitazone",
    # Common suffixes
    "pril", "sartan", "olol", "dipine", "statin", "azole", "mycin", "cillin",
    "floxacin", "xaban", "prazole", "mab", "nib", "vir", "tide", "dronate",
    "pine", "pamil", "azem", "done", "morphone", "codone", "pam", "lam",
    "zepam", "zolam", "oxacin", "cycline", "thiazide", "semide", "tanide",
    # Insulin variations
    "sulin", "insulin",
}

# Common drug prefixes
DRUG_PREFIXES = {
    "hydro", "chlor", "meth", "eth", "prop", "acet", "benz", "phen",
    "carb", "amino", "sulfa", "cef", "amox", "ampho", "flu", "pred",
}

# Minimum length for a drug name to be considered valid
MIN_DRUG_NAME_LENGTH = 3

# Cache for known drugs
_known_drugs: Optional[Set[str]] = None


def _load_known_drugs() -> Set[str]:
    """Load all known drug names from the medications JSON and rules."""
    global _known_drugs
    
    if _known_drugs is not None:
        return _known_drugs
    
    drugs = set()
    
    # Load from JSON file
    try:
        if _MEDICATIONS_JSON_PATH.exists():
            with open(_MEDICATIONS_JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract drugs from diabetes_medications
            if "diabetes_medications" in data:
                for class_data in data["diabetes_medications"].values():
                    for drug in class_data.get("drugs", []):
                        drugs.add(drug.lower())
            
            # Extract drugs from common_comorbidity_medications
            if "common_comorbidity_medications" in data:
                for class_data in data["common_comorbidity_medications"].values():
                    for drug in class_data.get("drugs", []):
                        drugs.add(drug.lower())
            
            # Extract drugs from drugs_affecting_glucose
            if "drugs_affecting_glucose" in data:
                for effect_list in data["drugs_affecting_glucose"].values():
                    for item in effect_list:
                        drug = item.get("drug", "")
                        if drug:
                            drugs.add(drug.lower())
            
            # Extract drugs from egfr_dosing_guidance
            if "egfr_dosing_guidance" in data:
                for drug in data["egfr_dosing_guidance"].keys():
                    drugs.add(drug.lower())
            
            logger.info(f"Loaded {len(drugs)} known drugs from medications JSON")
    except Exception as e:
        logger.error(f"Error loading known drugs from JSON: {e}")
    
    # Add common drugs from rules (hardcoded in rules.py)
    common_drugs = {
        # Common OTC and prescription drugs
        "aspirin", "ibuprofen", "naproxen", "acetaminophen", "tylenol", "advil",
        "lisinopril", "enalapril", "ramipril", "losartan", "valsartan",
        "amlodipine", "nifedipine", "diltiazem", "verapamil",
        "metoprolol", "atenolol", "carvedilol", "propranolol", "bisoprolol",
        "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin",
        "omeprazole", "pantoprazole", "esomeprazole", "lansoprazole",
        "sertraline", "fluoxetine", "paroxetine", "escitalopram", "citalopram",
        "gabapentin", "pregabalin", "duloxetine", "amitriptyline",
        "levothyroxine", "synthroid",
        "albuterol", "fluticasone", "montelukast",
        "warfarin", "apixaban", "rivaroxaban", "dabigatran", "enoxaparin",
        "ciprofloxacin", "levofloxacin", "azithromycin", "amoxicillin",
        "prednisone", "dexamethasone", "hydrocortisone", "methylprednisolone",
        "morphine", "oxycodone", "hydrocodone", "tramadol", "fentanyl",
        "furosemide", "hydrochlorothiazide", "spironolactone", "chlorthalidone",
        "allopurinol", "colchicine", "febuxostat",
        "methotrexate", "adalimumab", "infliximab", "etanercept",
        "sildenafil", "tadalafil",
        "montelukast", "cetirizine", "loratadine", "diphenhydramine",
        "zolpidem", "trazodone", "melatonin",
        # Vitamins and supplements (common queries)
        "vitamin d", "vitamin b12", "folic acid", "iron", "calcium",
    }
    drugs.update(common_drugs)
    
    _known_drugs = drugs
    logger.info(f"Total known drugs: {len(_known_drugs)}")
    return _known_drugs


def _has_drug_like_pattern(name: str) -> bool:
    """Check if a name has drug-like patterns (suffixes/prefixes)."""
    name_lower = name.lower()
    
    # Check for common drug suffixes
    for suffix in DRUG_SUFFIXES:
        if name_lower.endswith(suffix):
            return True
    
    # Check for common drug prefixes
    for prefix in DRUG_PREFIXES:
        if name_lower.startswith(prefix):
            return True
    
    return False


def _is_gibberish(name: str) -> bool:
    """Check if a name looks like gibberish/random text."""
    name_lower = name.lower().strip()
    
    # Too short
    if len(name_lower) < MIN_DRUG_NAME_LENGTH:
        return True
    
    # Only numbers
    if name_lower.isdigit():
        return True
    
    # Contains only repeated characters
    if len(set(name_lower.replace(" ", ""))) <= 2:
        return True
    
    # Common nonsense patterns
    nonsense_patterns = [
        r'^[a-z]{1,2}$',  # Single or double letters
        r'^(lol|lmao|wtf|omg|bruh|test|asdf|qwerty|hello|hi|hey|yo|sup)$',
        r'^[0-9]+$',  # Just numbers
        r'^[^a-zA-Z0-9]+$',  # Just special characters
        r'^(.)\1+$',  # Repeated single character
    ]
    
    for pattern in nonsense_patterns:
        if re.match(pattern, name_lower, re.IGNORECASE):
            return True
    
    # Check vowel ratio - real drug names have vowels
    vowels = sum(1 for c in name_lower if c in 'aeiou')
    if len(name_lower) > 4 and vowels == 0:
        return True
    
    return False


def validate_drug_name(drug_name: str) -> Tuple[bool, str]:
    """
    Validate if a drug name is a recognized medication.
    
    Returns:
        Tuple of (is_valid, reason)
        - is_valid: True if the drug name appears to be a valid medication
        - reason: Explanation of why validation failed (empty if valid)
    """
    if not drug_name or not drug_name.strip():
        return False, "Empty drug name provided"
    
    name = drug_name.strip()
    name_lower = name.lower()
    
    # Check for obvious gibberish first
    if _is_gibberish(name):
        return False, f"'{name}' does not appear to be a valid drug name"
    
    # Check if it's in our known drugs database
    known_drugs = _load_known_drugs()
    if name_lower in known_drugs:
        return True, ""
    
    # Check for partial matches (e.g., "aspirin 81mg" should match "aspirin")
    for known in known_drugs:
        if known in name_lower or name_lower in known:
            return True, ""
    
    # Check if it has drug-like patterns (common pharmaceutical suffixes)
    if _has_drug_like_pattern(name):
        return True, ""
    
    # If none of the above, flag as unknown
    return False, f"'{name}' is not recognized as a known medication. Please check the spelling or use the generic drug name."


def is_valid_drug(drug_name: str) -> bool:
    """Simple check if drug name is valid."""
    is_valid, _ = validate_drug_name(drug_name)
    return is_valid


def get_validation_error(drug_name: str) -> Optional[str]:
    """Get validation error message, or None if valid."""
    is_valid, reason = validate_drug_name(drug_name)
    return None if is_valid else reason
