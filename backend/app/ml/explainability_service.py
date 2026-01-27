"""
ML Explainability Service for Drug-Drug Interaction Predictions.

Provides SHAP and LIME explanations for model predictions to enhance
transparency and trust in the ML decision-making process.

Features:
- SHAP TreeExplainer for tree-based models (XGBoost, LightGBM, RF)
- LIME for local interpretable explanations
- Natural language explanation generation
- Feature importance visualization data
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")


# =============================================================================
# Feature Names (must match predictor.py feature extraction)
# =============================================================================

FEATURE_NAMES = [
    "drug1_matched", "drug2_matched",
    *[f"drug1_class_hash_{i}" for i in range(50)],
    *[f"drug2_class_hash_{i}" for i in range(50)],
    *[f"drug1_mechanism_hash_{i}" for i in range(30)],
    *[f"drug2_mechanism_hash_{i}" for i in range(30)],
    *[f"drug1_name_hash_{i}" for i in range(40)],
    *[f"drug2_name_hash_{i}" for i in range(40)],
]

# Human-readable feature group names
FEATURE_GROUPS = {
    "drug1_matched": "Drug 1 Database Match",
    "drug2_matched": "Drug 2 Database Match",
    "drug1_class": "Drug 1 Drug Class",
    "drug2_class": "Drug 2 Drug Class",
    "drug1_mechanism": "Drug 1 Mechanism of Action",
    "drug2_mechanism": "Drug 2 Mechanism of Action",
    "drug1_name": "Drug 1 Name Encoding",
    "drug2_name": "Drug 2 Name Encoding",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeatureContribution:
    """Single feature's contribution to the prediction."""
    feature_name: str
    feature_group: str
    value: float
    contribution: float  # SHAP value or LIME weight
    direction: str  # "positive" (increases risk) or "negative" (decreases risk)


@dataclass
class ExplanationResult:
    """Container for explanation results."""
    drug1_name: str
    drug2_name: str
    prediction_probability: float
    severity: str
    
    # Top contributing features
    top_positive_features: List[FeatureContribution]
    top_negative_features: List[FeatureContribution]
    
    # Summary statistics
    feature_importance_summary: Dict[str, float]  # Grouped by category
    
    # Natural language explanation
    natural_language_explanation: str
    
    # Method used
    explanation_method: str  # "shap" or "lime"
    
    # Raw data for visualization
    all_shap_values: Optional[List[float]] = None
    base_value: Optional[float] = None
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "drug1": self.drug1_name,
            "drug2": self.drug2_name,
            "prediction_probability": round(self.prediction_probability, 4),
            "severity": self.severity,
            "top_positive_features": [
                {
                    "name": f.feature_name,
                    "group": f.feature_group,
                    "contribution": round(f.contribution, 4),
                    "direction": f.direction,
                }
                for f in self.top_positive_features
            ],
            "top_negative_features": [
                {
                    "name": f.feature_name,
                    "group": f.feature_group,
                    "contribution": round(abs(f.contribution), 4),
                    "direction": f.direction,
                }
                for f in self.top_negative_features
            ],
            "feature_importance_summary": {
                k: round(v, 4) for k, v in self.feature_importance_summary.items()
            },
            "natural_language_explanation": self.natural_language_explanation,
            "explanation_method": self.explanation_method,
            "waterfall_data": {
                "shap_values": [round(v, 4) for v in (self.all_shap_values or [])],
                "base_value": round(self.base_value, 4) if self.base_value else None,
            } if self.all_shap_values else None,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# SHAP Explainer
# =============================================================================

class SHAPExplainer:
    """
    SHAP-based explainer for tree-based DDI prediction models.
    
    Uses TreeExplainer for fast, exact SHAP values on tree ensembles.
    """
    
    def __init__(self, models: Dict[str, Any]):
        """
        Initialize SHAP explainer with trained models.
        
        Args:
            models: Dictionary of model_name -> model object
        """
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP is not installed. Install with: pip install shap")
        
        self.models = models
        self.explainers: Dict[str, shap.TreeExplainer] = {}
        self._initialize_explainers()
    
    def _initialize_explainers(self) -> None:
        """Initialize SHAP TreeExplainer for each model."""
        for model_name, model_data in self.models.items():
            try:
                model = model_data.get('model', model_data) if isinstance(model_data, dict) else model_data
                
                # Handle CalibratedClassifierCV wrapper
                if hasattr(model, 'estimator'):
                    base_model = model.estimator
                elif hasattr(model, 'base_estimator'):
                    base_model = model.base_estimator
                elif hasattr(model, 'calibrated_classifiers_'):
                    # For CalibratedClassifierCV, use the first calibrated classifier's base
                    base_model = model.calibrated_classifiers_[0].estimator
                else:
                    base_model = model
                
                # Create TreeExplainer
                self.explainers[model_name] = shap.TreeExplainer(base_model)
                logger.info(f"Initialized SHAP TreeExplainer for {model_name}")
                
            except Exception as e:
                logger.warning(f"Could not create SHAP explainer for {model_name}: {e}")
    
    def explain(
        self,
        features: npt.NDArray[np.float32],
        drug1_name: str,
        drug2_name: str,
        prediction_probability: float,
        severity: str,
        top_k: int = 5
    ) -> ExplanationResult:
        """
        Generate SHAP explanation for a prediction.
        
        Args:
            features: Feature array (1, n_features)
            drug1_name: Name of first drug
            drug2_name: Name of second drug
            prediction_probability: Model's predicted probability
            severity: Predicted severity level
            top_k: Number of top features to highlight
            
        Returns:
            ExplanationResult with SHAP values and explanations
        """
        # Get SHAP values from first available explainer
        shap_values = None
        base_value = None
        explainer_used = None
        
        for model_name, explainer in self.explainers.items():
            try:
                X = features.reshape(1, -1) if features.ndim == 1 else features
                sv = explainer.shap_values(X)
                
                # Handle different SHAP value formats
                if isinstance(sv, list):
                    # For binary classification, use class 1 values
                    shap_values = sv[1][0] if len(sv) > 1 else sv[0][0]
                else:
                    shap_values = sv[0]
                
                base_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
                explainer_used = model_name
                break
                
            except Exception as e:
                logger.warning(f"SHAP explanation failed for {model_name}: {e}")
                continue
        
        if shap_values is None:
            raise RuntimeError("All SHAP explanations failed")
        
        # Process SHAP values
        return self._process_shap_values(
            shap_values=shap_values,
            base_value=base_value,
            drug1_name=drug1_name,
            drug2_name=drug2_name,
            prediction_probability=prediction_probability,
            severity=severity,
            top_k=top_k,
        )
    
    def _process_shap_values(
        self,
        shap_values: npt.NDArray,
        base_value: float,
        drug1_name: str,
        drug2_name: str,
        prediction_probability: float,
        severity: str,
        top_k: int
    ) -> ExplanationResult:
        """Process raw SHAP values into human-readable explanations."""
        
        # Create feature contributions
        contributions: List[FeatureContribution] = []
        
        for i, (name, value) in enumerate(zip(FEATURE_NAMES, shap_values)):
            group = self._get_feature_group(name)
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_group=group,
                value=float(value),
                contribution=float(value),
                direction="positive" if value > 0 else "negative"
            ))
        
        # Sort by absolute contribution
        sorted_contributions = sorted(contributions, key=lambda x: abs(x.contribution), reverse=True)
        
        # Get top positive and negative
        top_positive = [c for c in sorted_contributions if c.contribution > 0][:top_k]
        top_negative = [c for c in sorted_contributions if c.contribution < 0][:top_k]
        
        # Aggregate by feature group
        group_importance = self._aggregate_by_group(contributions)
        
        # Generate natural language explanation
        nl_explanation = self._generate_natural_language(
            drug1_name, drug2_name, prediction_probability, severity,
            top_positive, top_negative, group_importance
        )
        
        return ExplanationResult(
            drug1_name=drug1_name,
            drug2_name=drug2_name,
            prediction_probability=prediction_probability,
            severity=severity,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            feature_importance_summary=group_importance,
            natural_language_explanation=nl_explanation,
            explanation_method="shap",
            all_shap_values=shap_values.tolist(),
            base_value=base_value,
        )
    
    def _get_feature_group(self, feature_name: str) -> str:
        """Map feature name to human-readable group."""
        if feature_name == "drug1_matched":
            return FEATURE_GROUPS["drug1_matched"]
        elif feature_name == "drug2_matched":
            return FEATURE_GROUPS["drug2_matched"]
        elif "drug1_class" in feature_name:
            return FEATURE_GROUPS["drug1_class"]
        elif "drug2_class" in feature_name:
            return FEATURE_GROUPS["drug2_class"]
        elif "drug1_mechanism" in feature_name:
            return FEATURE_GROUPS["drug1_mechanism"]
        elif "drug2_mechanism" in feature_name:
            return FEATURE_GROUPS["drug2_mechanism"]
        elif "drug1_name" in feature_name:
            return FEATURE_GROUPS["drug1_name"]
        elif "drug2_name" in feature_name:
            return FEATURE_GROUPS["drug2_name"]
        return "Other"
    
    def _aggregate_by_group(self, contributions: List[FeatureContribution]) -> Dict[str, float]:
        """Aggregate contributions by feature group."""
        group_totals: Dict[str, float] = {}
        
        for c in contributions:
            group = c.feature_group
            group_totals[group] = group_totals.get(group, 0) + abs(c.contribution)
        
        # Normalize to percentages
        total = sum(group_totals.values()) or 1
        return {k: (v / total) * 100 for k, v in sorted(group_totals.items(), key=lambda x: -x[1])}
    
    def _generate_natural_language(
        self,
        drug1_name: str,
        drug2_name: str,
        probability: float,
        severity: str,
        top_positive: List[FeatureContribution],
        top_negative: List[FeatureContribution],
        group_importance: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation of the prediction."""
        
        # Start with summary
        risk_level = "high" if probability >= 0.7 else "moderate" if probability >= 0.4 else "low"
        
        explanation = f"The model predicts a {risk_level} risk ({probability:.0%}) of interaction between {drug1_name} and {drug2_name}. "
        
        # Add top contributing factors
        if top_positive:
            top_group = top_positive[0].feature_group
            explanation += f"The main factor increasing risk is the {top_group.lower()}. "
        
        # Add protective factors
        if top_negative:
            top_neg_group = top_negative[0].feature_group
            explanation += f"Some factors reducing the predicted risk include {top_neg_group.lower()}. "
        
        # Add group breakdown
        if group_importance:
            top_groups = list(group_importance.items())[:3]
            groups_text = ", ".join([f"{g[0]} ({g[1]:.0f}%)" for g in top_groups])
            explanation += f"Feature importance breakdown: {groups_text}."
        
        return explanation


# =============================================================================
# LIME Explainer
# =============================================================================

class LIMEExplainer:
    """
    LIME-based explainer for DDI prediction models.
    
    Provides local interpretable explanations using perturbation-based approach.
    """
    
    def __init__(self, models: Dict[str, Any], training_data: Optional[npt.NDArray] = None):
        """
        Initialize LIME explainer.
        
        Args:
            models: Dictionary of model_name -> model object
            training_data: Optional background data for better explanations
        """
        if not LIME_AVAILABLE:
            raise RuntimeError("LIME is not installed. Install with: pip install lime")
        
        self.models = models
        self.training_data = training_data
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self) -> None:
        """Initialize LIME tabular explainer."""
        # Create synthetic training data if not provided
        if self.training_data is None:
            # Generate random background data
            np.random.seed(42)
            self.training_data = np.random.rand(100, 242).astype(np.float32)
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=FEATURE_NAMES,
            class_names=['No Interaction', 'Interaction'],
            mode='classification',
            discretize_continuous=True,
        )
        logger.info("Initialized LIME TabularExplainer")
    
    def explain(
        self,
        features: npt.NDArray[np.float32],
        drug1_name: str,
        drug2_name: str,
        prediction_probability: float,
        severity: str,
        top_k: int = 5
    ) -> ExplanationResult:
        """
        Generate LIME explanation for a prediction.
        
        Args:
            features: Feature array
            drug1_name: Name of first drug
            drug2_name: Name of second drug
            prediction_probability: Model's predicted probability
            severity: Predicted severity level
            top_k: Number of top features to return
            
        Returns:
            ExplanationResult with LIME weights and explanations
        """
        # Get first available model for explanation
        predict_fn = None
        for model_name, model_data in self.models.items():
            try:
                model = model_data.get('model', model_data) if isinstance(model_data, dict) else model_data
                predict_fn = lambda x: model.predict_proba(x)
                break
            except Exception:
                continue
        
        if predict_fn is None:
            raise RuntimeError("No model available for LIME explanation")
        
        # Generate explanation
        X = features.reshape(1, -1)[0] if features.ndim == 2 else features
        exp = self.explainer.explain_instance(
            X,
            predict_fn,
            num_features=len(FEATURE_NAMES),
            top_labels=1,
        )
        
        # Extract feature weights
        feature_weights = exp.as_list(label=1)
        
        # Process into contributions
        contributions = []
        for feature_desc, weight in feature_weights:
            # Parse feature name from LIME's format
            feature_name = feature_desc.split()[0] if ' ' in feature_desc else feature_desc
            
            # Find matching feature name
            matched_name = next((f for f in FEATURE_NAMES if f in feature_desc), feature_desc)
            group = self._get_feature_group(matched_name)
            
            contributions.append(FeatureContribution(
                feature_name=matched_name,
                feature_group=group,
                value=0,  # LIME doesn't provide raw values easily
                contribution=weight,
                direction="positive" if weight > 0 else "negative"
            ))
        
        # Sort and filter
        top_positive = [c for c in contributions if c.contribution > 0][:top_k]
        top_negative = [c for c in contributions if c.contribution < 0][:top_k]
        
        # Aggregate by group
        group_importance = self._aggregate_by_group(contributions)
        
        # Generate explanation
        nl_explanation = self._generate_natural_language(
            drug1_name, drug2_name, prediction_probability, severity,
            top_positive, top_negative
        )
        
        return ExplanationResult(
            drug1_name=drug1_name,
            drug2_name=drug2_name,
            prediction_probability=prediction_probability,
            severity=severity,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            feature_importance_summary=group_importance,
            natural_language_explanation=nl_explanation,
            explanation_method="lime",
        )
    
    def _get_feature_group(self, feature_name: str) -> str:
        """Map feature name to human-readable group."""
        # Reuse same logic as SHAP
        if "class" in feature_name.lower():
            return "Drug Class"
        elif "mechanism" in feature_name.lower():
            return "Mechanism of Action"
        elif "matched" in feature_name.lower():
            return "Database Match"
        elif "name" in feature_name.lower():
            return "Drug Name Encoding"
        return "Other"
    
    def _aggregate_by_group(self, contributions: List[FeatureContribution]) -> Dict[str, float]:
        """Aggregate contributions by feature group."""
        group_totals: Dict[str, float] = {}
        
        for c in contributions:
            group = c.feature_group
            group_totals[group] = group_totals.get(group, 0) + abs(c.contribution)
        
        total = sum(group_totals.values()) or 1
        return {k: (v / total) * 100 for k, v in sorted(group_totals.items(), key=lambda x: -x[1])}
    
    def _generate_natural_language(
        self,
        drug1_name: str,
        drug2_name: str,
        probability: float,
        severity: str,
        top_positive: List[FeatureContribution],
        top_negative: List[FeatureContribution]
    ) -> str:
        """Generate human-readable explanation."""
        risk_level = "high" if probability >= 0.7 else "moderate" if probability >= 0.4 else "low"
        
        explanation = f"LIME analysis shows a {risk_level} interaction risk ({probability:.0%}) between {drug1_name} and {drug2_name}. "
        
        if top_positive:
            factors = [c.feature_group for c in top_positive[:2]]
            explanation += f"Key risk factors: {', '.join(factors)}. "
        
        if top_negative:
            factors = [c.feature_group for c in top_negative[:2]]
            explanation += f"Protective factors: {', '.join(factors)}."
        
        return explanation


# =============================================================================
# Unified Explainability Service
# =============================================================================

class ExplainabilityService:
    """
    Unified explainability service supporting both SHAP and LIME.
    
    Automatically selects the best available method and provides
    consistent explanation interface.
    """
    
    def __init__(self, models: Dict[str, Any]):
        """
        Initialize explainability service.
        
        Args:
            models: Dictionary of trained models
        """
        self.models = models
        self.shap_explainer: Optional[SHAPExplainer] = None
        self.lime_explainer: Optional[LIMEExplainer] = None
        
        # Initialize available explainers
        if SHAP_AVAILABLE:
            try:
                self.shap_explainer = SHAPExplainer(models)
                logger.info("SHAP explainer initialized")
            except Exception as e:
                logger.warning(f"SHAP initialization failed: {e}")
        
        if LIME_AVAILABLE:
            try:
                self.lime_explainer = LIMEExplainer(models)
                logger.info("LIME explainer initialized")
            except Exception as e:
                logger.warning(f"LIME initialization failed: {e}")
    
    def explain(
        self,
        features: npt.NDArray[np.float32],
        drug1_name: str,
        drug2_name: str,
        prediction_probability: float,
        severity: str,
        method: str = "auto",
        top_k: int = 5
    ) -> ExplanationResult:
        """
        Generate explanation for a prediction.
        
        Args:
            features: Feature array from prediction
            drug1_name: Name of first drug
            drug2_name: Name of second drug  
            prediction_probability: Model's prediction
            severity: Predicted severity
            method: "shap", "lime", or "auto" (tries SHAP first)
            top_k: Number of top features to return
            
        Returns:
            ExplanationResult with feature attributions and explanation
        """
        if method == "auto":
            # Prefer SHAP for tree models (faster, exact)
            if self.shap_explainer:
                method = "shap"
            elif self.lime_explainer:
                method = "lime"
            else:
                raise RuntimeError("No explainability method available")
        
        if method == "shap":
            if not self.shap_explainer:
                raise RuntimeError("SHAP not available")
            return self.shap_explainer.explain(
                features, drug1_name, drug2_name,
                prediction_probability, severity, top_k
            )
        
        elif method == "lime":
            if not self.lime_explainer:
                raise RuntimeError("LIME not available")
            return self.lime_explainer.explain(
                features, drug1_name, drug2_name,
                prediction_probability, severity, top_k
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_available_methods(self) -> List[str]:
        """Get list of available explanation methods."""
        methods = []
        if self.shap_explainer:
            methods.append("shap")
        if self.lime_explainer:
            methods.append("lime")
        return methods
    
    def is_available(self) -> bool:
        """Check if any explanation method is available."""
        return bool(self.shap_explainer or self.lime_explainer)


# =============================================================================
# Module-level instance
# =============================================================================

_explainability_service: Optional[ExplainabilityService] = None


def get_explainability_service(models: Dict[str, Any]) -> ExplainabilityService:
    """
    Get or create the global explainability service.
    
    Args:
        models: Trained model dictionary
        
    Returns:
        ExplainabilityService instance
    """
    global _explainability_service
    
    if _explainability_service is None:
        _explainability_service = ExplainabilityService(models)
    
    return _explainability_service
