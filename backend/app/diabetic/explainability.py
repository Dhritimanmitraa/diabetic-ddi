"""
SHAP-based Explainability for Drug Risk Predictions.

Provides feature attribution explanations for ML model predictions,
showing which patient factors contributed most to the risk assessment.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Feature name mapping to human-readable descriptions
FEATURE_DESCRIPTIONS = {
    "age": "Patient age",
    "gender_female": "Female patient",
    "gender_male": "Male patient",
    "creatinine": "Creatinine level",
    "potassium": "Potassium level",
    "fasting_glucose": "Fasting blood glucose",
    "has_nephropathy": "Kidney disease (nephropathy)",
    "has_retinopathy": "Eye complications (retinopathy)",
    "has_neuropathy": "Nerve damage (neuropathy)",
    "has_cardiovascular": "Heart/cardiovascular disease",
    "has_hypertension": "High blood pressure",
    "has_hyperlipidemia": "High cholesterol",
    "has_obesity": "Obesity",
    "drug_hash": "Drug chemical properties",
}

# Clinical interpretation for features
FEATURE_CLINICAL_IMPACT = {
    "age": "older patients have increased drug sensitivity",
    "creatinine": "indicates kidney function affecting drug clearance",
    "potassium": "affects cardiac safety with many drugs",
    "fasting_glucose": "indicates diabetes control status",
    "has_nephropathy": "impairs drug elimination",
    "has_retinopathy": "indicates advanced diabetes",
    "has_neuropathy": "may mask hypoglycemia symptoms",
    "has_cardiovascular": "increases risk with cardioactive drugs",
    "has_hypertension": "affects drug selection and dosing",
}


@dataclass
class FeatureAttribution:
    """Single feature's contribution to prediction."""
    feature_name: str
    feature_value: float
    shap_value: float
    description: str
    clinical_impact: str
    direction: str  # "increases_risk" or "decreases_risk"
    
    def to_dict(self) -> Dict:
        return {
            "feature": self.feature_name,
            "value": round(self.feature_value, 3) if isinstance(self.feature_value, float) else self.feature_value,
            "contribution": round(self.shap_value, 4),
            "description": self.description,
            "clinical_impact": self.clinical_impact,
            "direction": self.direction,
        }


@dataclass  
class SHAPExplanation:
    """Complete SHAP explanation for a prediction."""
    top_factors: List[FeatureAttribution]
    base_value: float
    prediction_value: float
    explanation_text: str
    
    def to_dict(self) -> Dict:
        return {
            "top_factors": [f.to_dict() for f in self.top_factors],
            "base_value": round(self.base_value, 4),
            "prediction_value": round(self.prediction_value, 4),
            "explanation_text": self.explanation_text,
        }


class SHAPExplainer:
    """
    SHAP-based explainer for drug risk prediction models.
    
    Uses TreeExplainer for tree-based models (XGBoost, RandomForest, LightGBM).
    Provides human-readable explanations of feature contributions.
    """
    
    # Feature names in order they appear in the model input vector
    NUMERIC_FEATURES = [
        "age", "gender_female", "gender_male", "creatinine", "potassium",
        "fasting_glucose", "has_nephropathy", "has_retinopathy", "has_neuropathy",
        "has_cardiovascular", "has_hypertension", "has_hyperlipidemia", "has_obesity"
    ]
    
    def __init__(self, model=None, hash_size: int = 48):
        self.model = model
        self.hash_size = hash_size
        self.explainer = None
        self._lock = threading.Lock()
        self._initialized = False
        
    def initialize(self, model) -> bool:
        """Initialize SHAP explainer with the model."""
        if self._initialized and self.model is model:
            return True
            
        try:
            import shap
            with self._lock:
                self.model = model
                # Use TreeExplainer for tree-based models
                self.explainer = shap.TreeExplainer(model)
                self._initialized = True
                logger.info("SHAP TreeExplainer initialized successfully")
                return True
        except ImportError:
            logger.warning("SHAP not installed. Run: pip install shap")
            return False
        except Exception as e:
            # NumPy 2.0 compatibility issue with SHAP - this is expected and handled gracefully
            if "NumPy" in str(e) or "numpy" in str(e).lower():
                logger.debug(f"SHAP not available due to NumPy compatibility: {e}")
            else:
                logger.error(f"Failed to initialize SHAP explainer: {e}")
            return False
    
    def explain_prediction(
        self,
        feature_vector: np.ndarray,
        patient_context: Dict,
        top_n: int = 5
    ) -> Optional[SHAPExplanation]:
        """
        Generate SHAP explanation for a prediction.
        
        Args:
            feature_vector: The input feature vector (1, n_features)
            patient_context: Original patient data for readable values
            top_n: Number of top contributing features to return
            
        Returns:
            SHAPExplanation with top factors and explanation text
        """
        if not self._initialized or self.explainer is None:
            return None
            
        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(feature_vector)
            
            # Handle multi-class output (take the class with highest probability)
            if isinstance(shap_values, list):
                # Get prediction to determine which class to explain
                try:
                    if hasattr(self.model, 'predict_proba'):
                        pred_proba = self.model.predict_proba(feature_vector)[0]
                        pred_class = int(np.argmax(pred_proba))
                    else:
                        # Fallback for regressors
                        pred = self.model.predict(feature_vector)[0]
                        pred_class = int(pred) if isinstance(pred, (int, np.integer)) else 0
                except (ValueError, AttributeError, TypeError):
                    # If predict_proba fails, use predict
                    pred = self.model.predict(feature_vector)[0]
                    pred_class = int(pred) if isinstance(pred, (int, np.integer)) else 0
                    pred_class = min(pred_class, len(shap_values) - 1) if pred_class >= 0 else 0
                
                shap_vals = shap_values[pred_class][0] if pred_class < len(shap_values) else shap_values[0][0]
                base_value = self.explainer.expected_value[pred_class] if pred_class < len(self.explainer.expected_value) else self.explainer.expected_value[0]
            else:
                shap_vals = shap_values[0]
                base_value = self.explainer.expected_value
            
            # Only explain the numeric features (first 13)
            # Skip drug hash features as they're not interpretable
            attributions = []
            for i, feature_name in enumerate(self.NUMERIC_FEATURES):
                if i >= len(shap_vals):
                    break
                    
                shap_val = float(shap_vals[i])
                feature_val = float(feature_vector[0, i])
                
                # Get description and clinical impact
                description = FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
                clinical_impact = FEATURE_CLINICAL_IMPACT.get(feature_name, "")
                
                # Determine direction
                direction = "increases_risk" if shap_val > 0 else "decreases_risk"
                
                attributions.append(FeatureAttribution(
                    feature_name=feature_name,
                    feature_value=feature_val,
                    shap_value=shap_val,
                    description=description,
                    clinical_impact=clinical_impact,
                    direction=direction
                ))
            
            # Sort by absolute SHAP value and take top N
            attributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
            top_factors = attributions[:top_n]
            
            # Generate explanation text
            explanation_text = self._generate_explanation_text(top_factors, patient_context)
            
            # Calculate prediction value
            prediction_value = base_value + sum(shap_vals)
            
            return SHAPExplanation(
                top_factors=top_factors,
                base_value=float(base_value),
                prediction_value=float(prediction_value),
                explanation_text=explanation_text
            )
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return None
    
    def _generate_explanation_text(
        self, 
        top_factors: List[FeatureAttribution],
        patient_context: Dict
    ) -> str:
        """Generate human-readable explanation from top factors."""
        if not top_factors:
            return "Unable to determine key contributing factors."
        
        # Filter to only risk-increasing factors for the primary explanation
        risk_factors = [f for f in top_factors if f.direction == "increases_risk"]
        protective = [f for f in top_factors if f.direction == "decreases_risk"]
        
        parts = []
        
        if risk_factors:
            risk_text = "Risk factors: "
            factor_strs = []
            for f in risk_factors[:3]:
                if f.clinical_impact:
                    factor_strs.append(f"{f.description} ({f.clinical_impact})")
                else:
                    factor_strs.append(f.description)
            risk_text += ", ".join(factor_strs)
            parts.append(risk_text)
        
        if protective:
            prot_text = "Protective factors: "
            prot_strs = [f.description for f in protective[:2]]
            prot_text += ", ".join(prot_strs)
            parts.append(prot_text)
        
        return ". ".join(parts) + "." if parts else "No significant factors identified."


# Singleton instance
_shap_explainer: Optional[SHAPExplainer] = None
_explainer_lock = threading.Lock()


def get_shap_explainer() -> SHAPExplainer:
    """Get or create the singleton SHAP explainer."""
    global _shap_explainer
    if _shap_explainer is None:
        with _explainer_lock:
            if _shap_explainer is None:
                _shap_explainer = SHAPExplainer()
    return _shap_explainer
