"""
LLM-based Drug Risk Checker running in parallel with ML model.

Uses Ollama to analyze drug risks for diabetic patients. This runs alongside
the ML model to provide complementary insights. The LLM provides reasoning
and context-aware analysis, while ML provides statistical predictions.

Recommended models for RTX 3050 4GB + 16GB RAM:
- llama3.1:8b - Best quality (current default)
- llama3.2:3b - Faster, good balance
- phi3:mini - Very efficient
- qwen2.5:3b - Good reasoning
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_MODEL = os.environ.get("OLLAMA_DRUG_CHECK_MODEL", "llama3.1:8b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TIMEOUT_SECONDS = 60  # Increased timeout for larger model
MAX_CONCURRENT_REQUESTS = 2  # Reduced for 8B model to avoid OOM

# System prompt for drug risk analysis (optimized for 8B model - better reasoning)
SYSTEM_PROMPT = """You are an expert clinical pharmacist specializing in diabetes care. Analyze drug risks for diabetic patients with deep clinical reasoning.

Assessment Framework:
1. Diabetes-specific risks: hypoglycemia/hyperglycemia, weight effects, CV impact
2. Kidney function: eGFR-based dosing adjustments, nephrotoxicity
3. Drug-drug interactions: especially with diabetes medications (metformin, insulin, etc.)
4. Comorbidities: cardiovascular, hypertension, liver disease, obesity
5. Patient factors: age, diabetes type, complications (nephropathy, retinopathy, neuropathy)

Risk Level Guidelines:
- safe: No known diabetes-specific risks, standard monitoring
- caution: Minor risks, requires monitoring (e.g., glucose, kidney function)
- high_risk: Significant risks requiring close monitoring or dose adjustment
- contraindicated: Should generally be avoided (e.g., eGFR <30 with certain drugs)
- fatal: Life-threatening interaction or contraindication

Be thorough but concise. Provide specific, actionable concerns.

Output ONLY valid JSON (no markdown, no extra text):
{
  "risk_level": "safe|caution|high_risk|contraindicated|fatal",
  "risk_score": 0-100,
  "reasoning": "2-3 sentence clinical explanation",
  "key_concerns": ["specific concern 1", "specific concern 2"],
  "monitoring_needed": ["specific test 1", "specific test 2"]
}"""


@dataclass
class LLMDrugRiskResult:
    """Result from LLM drug risk analysis."""

    risk_level: str
    risk_score: float
    reasoning: str
    key_concerns: List[str]
    monitoring_needed: List[str]
    model_used: str
    was_fallback: bool

    def to_dict(self) -> Dict:
        return {
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "reasoning": self.reasoning,
            "key_concerns": self.key_concerns,
            "monitoring_needed": self.monitoring_needed,
            "model_used": self.model_used,
            "was_fallback": self.was_fallback,
        }


class LLMDrugChecker:
    """
    LLM-based drug risk checker running in parallel with ML model.

    Uses Ollama for local inference. Designed to work alongside ML predictions
    to provide complementary reasoning and context-aware analysis.
    """

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST):
        self.model = model
        self.host = host
        self._client = None
        self._available = None
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def _get_client(self):
        """Lazy initialization of Ollama client."""
        if self._client is None:
            try:
                import ollama

                self._client = ollama.AsyncClient(host=self.host)
                self._available = True
            except ImportError:
                logger.warning("Ollama not installed. Run: pip install ollama")
                self._available = False
            except Exception as e:
                logger.error(f"Failed to initialize Ollama client: {e}")
                self._available = False
        return self._client

    async def is_available(self) -> bool:
        """Check if Ollama is available and model is ready."""
        try:
            client = await self._get_client()
            if client is None:
                return False
            # Try to list models to verify connection
            await asyncio.wait_for(client.list(), timeout=5.0)
            return True
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    async def check_drug_risk(
        self,
        drug_name: str,
        patient_context: Dict,
        current_medications: Optional[List[str]] = None,
    ) -> LLMDrugRiskResult:
        """
        Check drug risk using LLM analysis.

        Args:
            drug_name: Name of the drug to check
            patient_context: Patient clinical profile (age, eGFR, complications, etc.)
            current_medications: List of current medications

        Returns:
            LLMDrugRiskResult with risk assessment
        """
        prompt = self._build_prompt(drug_name, patient_context, current_medications)

        # Use semaphore to limit concurrent requests
        async with self._semaphore:
            try:
                client = await self._get_client()
                if client and await self.is_available():
                    response = await asyncio.wait_for(
                        client.chat(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            options={
                                "temperature": 0.2,  # Slightly higher for better reasoning with 8B model
                                "top_p": 0.9,  # More diverse reasoning
                                "num_predict": 400,  # Allow more detailed analysis with larger model
                            },
                        ),
                        timeout=TIMEOUT_SECONDS,
                    )

                    text = response["message"]["content"].strip()
                    return self._parse_response(text)

            except asyncio.TimeoutError:
                logger.warning(f"LLM drug check timed out for {drug_name}")
            except Exception as e:
                logger.warning(f"LLM drug check failed for {drug_name}: {e}")

        # Fallback to conservative assessment
        return self._generate_fallback(drug_name, patient_context)

    async def check_multiple_drugs(
        self,
        drug_names: List[str],
        patient_context: Dict,
        current_medications: Optional[List[str]] = None,
    ) -> Dict[str, LLMDrugRiskResult]:
        """
        Check multiple drugs in parallel (with concurrency limit).

        Args:
            drug_names: List of drug names to check
            patient_context: Patient clinical profile
            current_medications: List of current medications

        Returns:
            Dictionary mapping drug names to LLMDrugRiskResult
        """
        tasks = [
            self.check_drug_risk(drug, patient_context, current_medications)
            for drug in drug_names
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        drug_results = {}
        for drug, result in zip(drug_names, results):
            if isinstance(result, Exception):
                logger.error(f"Error checking {drug}: {result}")
                drug_results[drug] = self._generate_fallback(drug, patient_context)
            else:
                drug_results[drug] = result

        return drug_results

    def _build_prompt(
        self,
        drug_name: str,
        patient_context: Dict,
        current_medications: Optional[List[str]] = None,
    ) -> str:
        """Build the prompt for LLM analysis (optimized for speed)."""
        # Build compact patient profile
        profile_parts = []
        if patient_context.get("age"):
            profile_parts.append(f"Age:{patient_context['age']}")
        if patient_context.get("diabetes_type"):
            profile_parts.append(f"DM:{patient_context['diabetes_type']}")
        if patient_context.get("egfr"):
            profile_parts.append(f"eGFR:{patient_context['egfr']}")
        if patient_context.get("hba1c"):
            profile_parts.append(f"HbA1c:{patient_context['hba1c']}%")

        complications = []
        if patient_context.get("has_nephropathy"):
            complications.append("Nephropathy")
        if patient_context.get("has_cardiovascular"):
            complications.append("CVD")
        if patient_context.get("has_neuropathy"):
            complications.append("Neuropathy")
        if patient_context.get("has_hypertension"):
            complications.append("HTN")

        prompt = f"Drug: {drug_name}\nPatient: {', '.join(profile_parts)}"
        if complications:
            prompt += f" | Complications: {', '.join(complications)}"
        if current_medications:
            meds_str = ", ".join(current_medications[:5])  # Limit to 5 meds
            prompt += f"\nCurrent meds: {meds_str}"

        prompt += "\n\nProvide JSON risk assessment only."

        return prompt

    def _parse_response(self, text: str) -> LLMDrugRiskResult:
        """Parse LLM response into structured result."""
        try:
            # Extract JSON from response
            text = text.strip()
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()

            # Parse JSON
            data = json.loads(text)

            return LLMDrugRiskResult(
                risk_level=data.get("risk_level", "caution").lower(),
                risk_score=float(data.get("risk_score", 50)),
                reasoning=data.get("reasoning", "LLM analysis completed"),
                key_concerns=data.get("key_concerns", []),
                monitoring_needed=data.get("monitoring_needed", []),
                model_used=self.model,
                was_fallback=False,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}, text: {text[:200]}")
            # Try to extract risk level from text
            risk_level = "caution"
            if "fatal" in text.lower() or "contraindicated" in text.lower():
                risk_level = "contraindicated"
            elif "high risk" in text.lower():
                risk_level = "high_risk"
            elif "safe" in text.lower():
                risk_level = "safe"

            return LLMDrugRiskResult(
                risk_level=risk_level,
                risk_score=50.0,
                reasoning=text[:200],
                key_concerns=[],
                monitoring_needed=[],
                model_used=self.model,
                was_fallback=True,
            )

    def _generate_fallback(
        self, drug_name: str, patient_context: Dict
    ) -> LLMDrugRiskResult:
        """Generate conservative fallback assessment."""
        egfr = patient_context.get("egfr")
        if egfr and egfr < 30:
            risk_level = "high_risk"
            risk_score = 75.0
            reasoning = f"Conservative assessment: {drug_name} requires caution given severe kidney impairment (eGFR {egfr})"
        elif egfr and egfr < 60:
            risk_level = "caution"
            risk_score = 50.0
            reasoning = f"Conservative assessment: {drug_name} may require dose adjustment given reduced kidney function (eGFR {egfr})"
        else:
            risk_level = "caution"
            risk_score = 40.0
            reasoning = f"Conservative assessment: {drug_name} appears generally safe but requires standard monitoring"

        return LLMDrugRiskResult(
            risk_level=risk_level,
            risk_score=risk_score,
            reasoning=reasoning,
            key_concerns=["Standard diabetic patient monitoring recommended"],
            monitoring_needed=["Monitor blood glucose", "Monitor kidney function"],
            model_used="fallback",
            was_fallback=True,
        )


# Singleton instance
_llm_checker: Optional[LLMDrugChecker] = None


def get_llm_checker() -> LLMDrugChecker:
    """Get or create the singleton LLM drug checker."""
    global _llm_checker
    # Check if we need to recreate (model changed or first time)
    if _llm_checker is None or _llm_checker.model != DEFAULT_MODEL:
        _llm_checker = LLMDrugChecker()
    return _llm_checker


def reset_llm_checker():
    """Reset the singleton instance (useful for testing or model changes)."""
    global _llm_checker
    _llm_checker = None


async def check_ollama_drug_checker_status() -> Dict:
    """Check Ollama availability for drug checking."""
    checker = get_llm_checker()
    available = await checker.is_available()

    return {
        "ollama_available": available,
        "model": checker.model,
        "host": checker.host,
        "fallback_enabled": True,
    }
