"""
LLM-based Explanation Generator for Drug Risk Assessments.

Uses Ollama for local LLM inference to generate patient-friendly
explanations of drug risk assessments. The LLM ONLY explains —
it NEVER makes medical decisions.

Recommended models for 16GB RAM:
- llama3.2:3b - Best balance of speed and quality (recommended)
- mistral:7b - Good alternative (if available)
- llama3.1:8b - Another option
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TIMEOUT_SECONDS = 45  # Increased timeout for larger model


# System prompt enforcing safety - LLM explains, never decides
SYSTEM_PROMPT = """You are a medical information assistant that explains drug risk assessments to patients in simple, friendly language.

CRITICAL SAFETY RULES:
1. You ONLY explain decisions that have already been made by validated clinical rules
2. You NEVER recommend or suggest changing medications
3. You NEVER contradict the risk assessment provided to you
4. You ALWAYS recommend consulting with their healthcare provider
5. Keep explanations under 100 words
6. Use simple language a patient can understand
7. Be reassuring but honest about risks

You will receive the drug name, risk level, and key factors. Your job is to explain WHY this risk level was assigned in a way patients can understand."""


@dataclass
class LLMExplanation:
    """Result from LLM explanation generation."""

    patient_friendly_text: str
    model_used: str
    was_fallback: bool  # True if template was used instead of LLM

    def to_dict(self) -> Dict:
        return {
            "text": self.patient_friendly_text,
            "model": self.model_used,
            "was_fallback": self.was_fallback,
        }


class LLMExplainer:
    """
    Ollama-based LLM explainer for patient-friendly explanations.

    SAFETY: The LLM only generates explanations for decisions already
    made by the rule-based system. It cannot override or modify decisions.
    """

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST):
        self.model = model
        self.host = host
        self._client = None
        self._available = None

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

    async def generate_explanation(
        self,
        drug_name: str,
        risk_level: str,
        risk_factors: List[str],
        recommendation: str,
        shap_factors: Optional[List[Dict]] = None,
    ) -> LLMExplanation:
        """
        Generate a patient-friendly explanation of the drug risk assessment.

        Args:
            drug_name: Name of the drug being assessed
            risk_level: The risk level (safe, caution, high_risk, etc.)
            risk_factors: List of clinical risk factors identified
            recommendation: The clinical recommendation
            shap_factors: Optional SHAP-based feature attributions

        Returns:
            LLMExplanation with patient-friendly text
        """
        # Build the prompt
        prompt = self._build_prompt(
            drug_name, risk_level, risk_factors, recommendation, shap_factors
        )

        # Try LLM first
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
                        options={"temperature": 0.3},  # Lower temp for consistency
                    ),
                    timeout=TIMEOUT_SECONDS,
                )

                text = response["message"]["content"].strip()
                return LLMExplanation(
                    patient_friendly_text=text,
                    model_used=self.model,
                    was_fallback=False,
                )

        except asyncio.TimeoutError:
            logger.warning(f"LLM generation timed out after {TIMEOUT_SECONDS}s")
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")

        # Fallback to template-based explanation
        fallback_text = self._generate_fallback(
            drug_name, risk_level, risk_factors, recommendation
        )
        return LLMExplanation(
            patient_friendly_text=fallback_text,
            model_used="template",
            was_fallback=True,
        )

    def _build_prompt(
        self,
        drug_name: str,
        risk_level: str,
        risk_factors: List[str],
        recommendation: str,
        shap_factors: Optional[List[Dict]] = None,
    ) -> str:
        """Build the prompt for the LLM."""
        prompt = f"""Explain this drug risk assessment to a patient:

Drug: {drug_name}
Risk Level: {risk_level.upper().replace('_', ' ')}
Key Factors: {', '.join(risk_factors[:5]) if risk_factors else 'General precaution'}
Clinical Recommendation: {recommendation}
"""

        if shap_factors:
            factor_strs = []
            for f in shap_factors[:3]:
                direction = (
                    "↑ risk" if f.get("direction") == "increases_risk" else "↓ risk"
                )
                factor_strs.append(f"• {f.get('description', '')}: {direction}")
            prompt += f"\nContributing Factors:\n" + "\n".join(factor_strs)

        prompt += "\n\nExplain this in simple, reassuring language for the patient. Remind them to discuss with their doctor."

        return prompt

    def _generate_fallback(
        self,
        drug_name: str,
        risk_level: str,
        risk_factors: List[str],
        recommendation: str,
    ) -> str:
        """Generate template-based fallback explanation."""
        risk_descriptions = {
            "safe": f"{drug_name} appears to be safe for you based on your health profile.",
            "caution": f"{drug_name} can be used, but requires some monitoring due to your health conditions.",
            "high_risk": f"{drug_name} carries some risks for you. Close monitoring will be needed.",
            "contraindicated": f"{drug_name} is not recommended for you due to potential serious risks.",
            "fatal": f"{drug_name} poses serious risks that could be life-threatening for you.",
        }

        base = risk_descriptions.get(
            risk_level, f"The risk level for {drug_name} is {risk_level}."
        )

        if risk_factors:
            factors_text = f" This is because of: {', '.join(risk_factors[:3])}."
        else:
            factors_text = ""

        closing = " Please discuss this with your healthcare provider before making any changes to your medications."

        return base + factors_text + closing


# Singleton instance
_llm_explainer: Optional[LLMExplainer] = None


def get_llm_explainer() -> LLMExplainer:
    """Get or create the singleton LLM explainer."""
    global _llm_explainer
    # Check if we need to recreate (model changed or first time)
    if _llm_explainer is None or _llm_explainer.model != DEFAULT_MODEL:
        _llm_explainer = LLMExplainer()
    return _llm_explainer


def reset_llm_explainer():
    """Reset the singleton instance (useful for testing or model changes)."""
    global _llm_explainer
    _llm_explainer = None


async def check_ollama_status() -> Dict:
    """Check Ollama availability and return status info."""
    explainer = get_llm_explainer()
    available = await explainer.is_available()

    return {
        "ollama_available": available,
        "model": explainer.model,
        "host": explainer.host,
        "fallback_enabled": True,
    }
