"""
Smart Model Selector for Diabetic Drug Risk Assessment.

Intelligently combines Rules, ML, and LLM predictions based on:
1. ML confidence levels
2. Rule-based risk floors for known drug classes
3. Agreement/disagreement between models
4. Clinical context (eGFR, complications, etc.)

This addresses the ML overconfidence problem by:
- Applying rule-based minimum risk floors
- Trusting LLM when ML confidence is low
- Using ensemble logic that prioritizes safety
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Rule-based minimum risk floors for known problematic drug classes
# These prevent ML from being overconfident about "safe" predictions
RULE_RISK_FLOORS = {
    # Corticosteroids - known to worsen glucose control
    "corticosteroid": 60,
    "prednisone": 60,
    "prednisolone": 60,
    "dexamethasone": 60,
    "methylprednisolone": 60,
    "hydrocortisone": 50,
    
    # Thiazide diuretics - known to worsen glucose control
    "thiazide": 40,
    "hydrochlorothiazide": 40,
    "chlorthalidone": 40,
    "indapamide": 40,
    
    # Antipsychotics - high risk for weight gain and glucose issues
    "antipsychotic": 50,
    "olanzapine": 60,
    "clozapine": 65,
    "quetiapine": 50,
    "risperidone": 45,
    
    # Beta-blockers (non-selective) - can mask hypoglycemia
    "beta_blocker": 30,
    "propranolol": 35,
    "nadolol": 35,
    
    # Statins - small glucose increase
    "statin": 20,
    
    # Loop diuretics - generally safer but still need monitoring
    "loop_diuretic": 25,
    "furosemide": 25,
    "bumetanide": 25,
    
    # NSAIDs - kidney risk in diabetes
    "nsaid": 35,
    "ibuprofen": 35,
    "naproxen": 35,
    "diclofenac": 35,
    
    # ACE inhibitors / ARBs - generally safe but hyperkalemia risk
    "ace_inhibitor": 15,
    "arb": 15,
}

# Drug class patterns (for fuzzy matching)
DRUG_CLASS_PATTERNS = {
    "corticosteroid": ["prednisone", "prednisolone", "dexamethasone", "methylprednisolone", "hydrocortisone", "cortisone", "betamethasone"],
    "thiazide": ["hydrochlorothiazide", "chlorthalidone", "indapamide", "metolazone"],
    "antipsychotic": ["olanzapine", "clozapine", "quetiapine", "risperidone", "aripiprazole", "ziprasidone"],
    "beta_blocker": ["propranolol", "nadolol", "timolol", "pindolol"],
    "nsaid": ["ibuprofen", "naproxen", "diclofenac", "indomethacin", "ketorolac"],
}


@dataclass
class SmartModelResult:
    """Result from smart model selection."""
    final_risk_level: str
    final_risk_score: float
    decision_source: str  # "rules", "ml_high_confidence", "llm_primary", "ensemble", "rule_floor"
    confidence: float  # 0-1, how confident we are in this prediction
    ml_risk_level: Optional[str] = None
    ml_risk_score: Optional[float] = None
    ml_confidence: Optional[float] = None
    llm_risk_level: Optional[str] = None
    llm_risk_score: Optional[float] = None
    rule_risk_level: Optional[str] = None
    rule_risk_score: Optional[float] = None
    applied_floor: Optional[Tuple[str, float]] = None  # (drug_class, floor_value)
    reasoning: str = ""


def get_drug_class(drug_name: str) -> Optional[str]:
    """Identify drug class from drug name (fuzzy matching)."""
    drug_lower = drug_name.lower()
    
    # Direct match
    if drug_lower in RULE_RISK_FLOORS:
        return drug_lower
    
    # Pattern matching
    for class_name, patterns in DRUG_CLASS_PATTERNS.items():
        for pattern in patterns:
            if pattern in drug_lower:
                return class_name
    
    return None


def apply_rule_floor(
    drug_name: str,
    ml_risk_score: float,
    ml_risk_level: str
) -> Tuple[float, Optional[Tuple[str, float]]]:
    """
    Apply rule-based minimum risk floor.
    
    Returns:
        (adjusted_risk_score, (drug_class, floor_value) or None)
    """
    drug_class = get_drug_class(drug_name)
    
    if drug_class and drug_class in RULE_RISK_FLOORS:
        floor_value = RULE_RISK_FLOORS[drug_class]
        if ml_risk_score < floor_value:
            logger.info(f"Applied rule floor: {drug_name} ({drug_class}) - ML said {ml_risk_score}, floor is {floor_value}")
            return floor_value, (drug_class, floor_value)
    
    return ml_risk_score, None


def select_smart_model(
    rule_risk_level: str,
    rule_risk_score: float,
    ml_risk_level: Optional[str] = None,
    ml_risk_score: Optional[float] = None,
    ml_confidence: Optional[float] = None,
    llm_risk_level: Optional[str] = None,
    llm_risk_score: Optional[float] = None,
    drug_name: str = "",
) -> SmartModelResult:
    """
    Intelligently select the best model prediction.
    
    Strategy:
    1. Rules are always the baseline (clinically validated)
    2. If ML confidence is high (>0.85) and agrees with rules → use ML
    3. If ML confidence is low (<0.7) → trust LLM if available, otherwise rules
    4. If ML says "safe" but rules say risk → apply rule floor
    5. If LLM disagrees with ML and ML confidence is low → trust LLM
    6. Always apply rule floors for known problematic drug classes
    
    Returns:
        SmartModelResult with final decision
    """
    # Start with rules as baseline
    final_risk_level = rule_risk_level
    final_risk_score = rule_risk_score
    decision_source = "rules"
    confidence = 0.8  # Rules are clinically validated
    reasoning_parts = []
    
    # Apply rule floor if ML exists and drug is in known problematic class
    applied_floor = None
    if ml_risk_score is not None:
        adjusted_score, floor_info = apply_rule_floor(drug_name, ml_risk_score, ml_risk_level or "safe")
        if floor_info:
            applied_floor = floor_info
            ml_risk_score = adjusted_score
            # Update ML risk level based on floor
            if adjusted_score >= 60:
                ml_risk_level = "high_risk"
            elif adjusted_score >= 40:
                ml_risk_level = "caution"
            reasoning_parts.append(f"Applied rule floor ({floor_info[0]}: {floor_info[1]})")
    
    # Decision logic
    if ml_risk_level is None:
        # No ML available - use rules
        reasoning_parts.append("ML unavailable, using rules")
        
    elif ml_confidence is None or ml_confidence < 0.7:
        # Low ML confidence - trust LLM if available, otherwise rules
        if llm_risk_level:
            final_risk_level = llm_risk_level
            final_risk_score = llm_risk_score or rule_risk_score
            decision_source = "llm_primary"
            confidence = 0.75
            reasoning_parts.append(f"ML confidence low ({ml_confidence:.2f}), using LLM")
        else:
            reasoning_parts.append(f"ML confidence low ({ml_confidence:.2f}), using rules")
            
    elif ml_confidence > 0.85:
        # High ML confidence
        if ml_risk_level == rule_risk_level:
            # Agreement - use ML (it's confident and agrees)
            final_risk_level = ml_risk_level
            final_risk_score = ml_risk_score or rule_risk_score
            decision_source = "ml_high_confidence"
            confidence = min(0.9, ml_confidence)
            reasoning_parts.append(f"ML high confidence ({ml_confidence:.2f}) and agrees with rules")
        else:
            # Disagreement - check if LLM can break tie
            if llm_risk_level:
                # Use LLM as tiebreaker
                if llm_risk_level == rule_risk_level:
                    # LLM agrees with rules - trust rules
                    decision_source = "rules_llm_agreement"
                    confidence = 0.85
                    reasoning_parts.append("ML disagrees, but LLM agrees with rules - trusting rules")
                elif llm_risk_level == ml_risk_level:
                    # LLM agrees with ML - use ML
                    final_risk_level = ml_risk_level
                    final_risk_score = ml_risk_score or rule_risk_score
                    decision_source = "ml_llm_agreement"
                    confidence = 0.8
                    reasoning_parts.append("ML and LLM agree, using ML")
                else:
                    # Three-way disagreement - use rules (safest)
                    decision_source = "rules_three_way_disagreement"
                    confidence = 0.7
                    reasoning_parts.append("Three-way disagreement, using rules (safest)")
            else:
                # No LLM - trust rules when ML disagrees
                decision_source = "rules_ml_disagreement"
                confidence = 0.75
                reasoning_parts.append(f"ML disagrees with rules, trusting rules (ML conf: {ml_confidence:.2f})")
    else:
        # Medium ML confidence (0.7-0.85)
        if ml_risk_level == rule_risk_level:
            # Agreement - use ML
            final_risk_level = ml_risk_level
            final_risk_score = ml_risk_score or rule_risk_score
            decision_source = "ml_medium_confidence"
            confidence = ml_confidence
            reasoning_parts.append(f"ML medium confidence ({ml_confidence:.2f}) and agrees with rules")
        else:
            # Disagreement - prefer rules
            if llm_risk_level and llm_risk_level == rule_risk_level:
                decision_source = "rules_llm_agreement"
                confidence = 0.8
                reasoning_parts.append("ML disagrees, LLM agrees with rules - trusting rules")
            else:
                decision_source = "rules_ml_disagreement"
                confidence = 0.7
                reasoning_parts.append(f"ML disagrees with rules, trusting rules")
    
    # Safety check: if final risk is "safe" but any model says risk, be cautious
    if final_risk_level == "safe":
        if rule_risk_level != "safe":
            final_risk_level = rule_risk_level
            final_risk_score = rule_risk_score
            decision_source = "rules_safety_override"
            reasoning_parts.append("Safety override: rules indicate risk")
        elif ml_risk_level and ml_risk_level != "safe" and ml_confidence and ml_confidence > 0.6:
            # ML says risk with decent confidence
            final_risk_level = ml_risk_level
            final_risk_score = ml_risk_score or rule_risk_score
            decision_source = "ml_safety_override"
            reasoning_parts.append("Safety override: ML indicates risk")
        elif llm_risk_level and llm_risk_level != "safe":
            # LLM says risk
            final_risk_level = llm_risk_level
            final_risk_score = llm_risk_score or rule_risk_score
            decision_source = "llm_safety_override"
            reasoning_parts.append("Safety override: LLM indicates risk")
    
    return SmartModelResult(
        final_risk_level=final_risk_level,
        final_risk_score=final_risk_score,
        decision_source=decision_source,
        confidence=confidence,
        ml_risk_level=ml_risk_level,
        ml_risk_score=ml_risk_score,
        ml_confidence=ml_confidence,
        llm_risk_level=llm_risk_level,
        llm_risk_score=llm_risk_score,
        rule_risk_level=rule_risk_level,
        rule_risk_score=rule_risk_score,
        applied_floor=applied_floor,
        reasoning="; ".join(reasoning_parts) if reasoning_parts else "Standard assessment"
    )

