"""
Prompt templates for LLM drug interaction analysis
"""

SYSTEM_PROMPT = """You are an expert clinical pharmacologist specializing in drug-drug interactions (DDI). 
Your role is to analyze potential interactions between medications and provide clear, evidence-based assessments.

IMPORTANT GUIDELINES:
1. Be conservative - when in doubt, err on the side of caution
2. Provide clear severity ratings: none, mild, moderate, severe, or contraindicated
3. Explain the mechanism of interaction when known
4. Always recommend consulting a healthcare provider for serious interactions
5. Consider pharmacokinetic (absorption, distribution, metabolism, excretion) and pharmacodynamic interactions

You must respond in a specific JSON format for parsing."""


def build_prediction_prompt(
    drug1: str, drug2: str, twosides_data: dict | None = None
) -> str:
    """
    Build the prediction prompt with optional TWOSIDES context
    """
    context_section = ""

    if twosides_data:
        context_section = f"""
## Known Database Information (TWOSIDES):
- Known interaction recorded: {twosides_data.get('known_interaction', 'Unknown')}
- Number of reported interactions: {twosides_data.get('interaction_count', 0)}
- Reported side effects: {', '.join(twosides_data.get('side_effects', [])[:10]) or 'None recorded'}
"""

    prompt = f"""Analyze the potential drug-drug interaction between the following medications:

## Drug Pair:
- **Drug 1**: {drug1}
- **Drug 2**: {drug2}
{context_section}

## Your Task:
Analyze whether these two drugs interact with each other. Consider:
1. Direct pharmacological interactions
2. Effects on drug metabolism (CYP450 enzymes)
3. Additive or synergistic effects
4. Contraindications

## Required Response Format (JSON):
{{
    "has_interaction": true/false,
    "severity": "none" | "mild" | "moderate" | "severe" | "contraindicated",
    "confidence": 0.0-1.0,
    "explanation": "Clear explanation of the interaction or lack thereof",
    "mechanism": "Pharmacological mechanism if interaction exists",
    "recommendations": ["List", "of", "clinical", "recommendations"],
    "reasoning": "Step-by-step reasoning for your assessment"
}}

Provide ONLY the JSON response, no additional text."""

    return prompt


FALLBACK_RESPONSE = {
    "has_interaction": False,
    "severity": "unknown",
    "confidence": 0.0,
    "explanation": "Unable to analyze interaction. Please consult a healthcare provider.",
    "mechanism": "Unknown",
    "recommendations": [
        "Consult a pharmacist or physician",
        "Review drug package inserts",
        "Check with your healthcare provider before combining medications",
    ],
    "reasoning": "LLM analysis was unable to complete successfully.",
}
