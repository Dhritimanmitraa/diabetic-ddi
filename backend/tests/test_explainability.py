"""
Tests for SHAP and LLM explainability modules.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock


class TestSHAPExplainer:
    """Tests for SHAP-based explainability."""

    def test_shap_explainer_initialization(self):
        """Test SHAP explainer can be created."""
        from app.diabetic.explainability import SHAPExplainer

        explainer = SHAPExplainer()
        assert explainer is not None
        assert not explainer._initialized

    def test_feature_descriptions_exist(self):
        """Test that feature descriptions are defined."""
        from app.diabetic.explainability import FEATURE_DESCRIPTIONS

        assert "age" in FEATURE_DESCRIPTIONS
        assert "has_nephropathy" in FEATURE_DESCRIPTIONS
        assert "potassium" in FEATURE_DESCRIPTIONS

    def test_clinical_impact_mapping(self):
        """Test clinical impact mappings exist."""
        from app.diabetic.explainability import FEATURE_CLINICAL_IMPACT

        assert (
            "egfr" not in FEATURE_CLINICAL_IMPACT
            or "kidney" in FEATURE_CLINICAL_IMPACT.get("egfr", "").lower()
            or "creatinine" in FEATURE_CLINICAL_IMPACT
        )
        assert "has_cardiovascular" in FEATURE_CLINICAL_IMPACT

    def test_feature_attribution_to_dict(self):
        """Test FeatureAttribution serialization."""
        from app.diabetic.explainability import FeatureAttribution

        attr = FeatureAttribution(
            feature_name="age",
            feature_value=70.0,
            shap_value=0.25,
            description="Patient age",
            clinical_impact="older patients have increased drug sensitivity",
            direction="increases_risk",
        )
        d = attr.to_dict()
        assert d["feature"] == "age"
        assert d["value"] == 70.0
        assert d["contribution"] == 0.25
        assert d["direction"] == "increases_risk"

    def test_shap_explanation_to_dict(self):
        """Test SHAPExplanation serialization."""
        from app.diabetic.explainability import SHAPExplanation, FeatureAttribution

        attr = FeatureAttribution(
            feature_name="age",
            feature_value=70.0,
            shap_value=0.25,
            description="Patient age",
            clinical_impact="test",
            direction="increases_risk",
        )
        explanation = SHAPExplanation(
            top_factors=[attr],
            base_value=0.5,
            prediction_value=0.75,
            explanation_text="Risk factors: Patient age (test).",
        )
        d = explanation.to_dict()
        assert len(d["top_factors"]) == 1
        assert "explanation_text" in d


class TestLLMExplainer:
    """Tests for LLM-based explanations."""

    def test_llm_explainer_initialization(self):
        """Test LLM explainer can be created."""
        from app.diabetic.llm_explainer import LLMExplainer

        explainer = LLMExplainer()
        assert explainer is not None
        assert explainer.model == "llama3.1:8b"

    def test_fallback_explanation_generation(self):
        """Test fallback templates work when LLM unavailable."""
        from app.diabetic.llm_explainer import LLMExplainer

        explainer = LLMExplainer()

        text = explainer._generate_fallback(
            drug_name="Metformin",
            risk_level="caution",
            risk_factors=["low kidney function", "age over 65"],
            recommendation="Monitor diabetic function",
        )

        assert "Metformin" in text
        assert "monitoring" in text.lower() or "caution" in text.lower()
        assert "healthcare provider" in text.lower()

    def test_fallback_for_fatal_risk(self):
        """Test fallback correctly handles fatal risks."""
        from app.diabetic.llm_explainer import LLMExplainer

        explainer = LLMExplainer()

        text = explainer._generate_fallback(
            drug_name="DrugX",
            risk_level="fatal",
            risk_factors=["severe interaction"],
            recommendation="Do not use",
        )

        assert "life-threatening" in text.lower() or "serious" in text.lower()

    def test_llm_explanation_to_dict(self):
        """Test LLMExplanation serialization."""
        from app.diabetic.llm_explainer import LLMExplanation

        result = LLMExplanation(
            patient_friendly_text="Test explanation",
            model_used="template",
            was_fallback=True,
        )
        d = result.to_dict()
        assert d["text"] == "Test explanation"
        assert d["was_fallback"] is True

    @pytest.mark.anyio
    async def test_generate_explanation_fallback(self):
        """Test that generate_explanation falls back gracefully."""
        from app.diabetic.llm_explainer import LLMExplainer

        explainer = LLMExplainer()
        explainer._available = False  # Force fallback

        result = await explainer.generate_explanation(
            drug_name="Lisinopril",
            risk_level="high_risk",
            risk_factors=["hyperkalemia risk"],
            recommendation="Monitor potassium",
        )

        assert result.was_fallback is True
        assert "Lisinopril" in result.patient_friendly_text


class TestIntegration:
    """Integration tests for explainability."""

    def test_singleton_shap_explainer(self):
        """Test SHAP explainer singleton pattern."""
        from app.diabetic.explainability import get_shap_explainer

        e1 = get_shap_explainer()
        e2 = get_shap_explainer()
        assert e1 is e2

    def test_singleton_llm_explainer(self):
        """Test LLM explainer singleton pattern."""
        from app.diabetic.llm_explainer import get_llm_explainer

        e1 = get_llm_explainer()
        e2 = get_llm_explainer()
        assert e1 is e2
