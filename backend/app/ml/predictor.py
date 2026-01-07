"""
Prediction Service for Drug-Drug Interaction.

Provides inference using trained ML models.
Compatible with both DDIModel format and direct sklearn models.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
import os
import json
import logging
from datetime import datetime, timezone
import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class PredictionResult:
    """Container for prediction results."""

    def __init__(
        self,
        drug1_name: str,
        drug2_name: str,
        interaction_probability: float,
        predicted_interaction: bool,
        severity_prediction: str,
        confidence: float,
        model_predictions: Dict[str, float],
        timestamp: datetime = None,
    ):
        self.drug1_name = drug1_name
        self.drug2_name = drug2_name
        self.interaction_probability = interaction_probability
        self.predicted_interaction = predicted_interaction
        self.severity_prediction = severity_prediction
        self.confidence = confidence
        self.model_predictions = model_predictions
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "drug1": self.drug1_name,
            "drug2": self.drug2_name,
            "interaction_probability": round(self.interaction_probability, 4),
            "predicted_interaction": self.predicted_interaction,
            "severity_prediction": self.severity_prediction,
            "confidence": round(self.confidence, 4),
            "model_predictions": {
                k: round(v, 4) for k, v in self.model_predictions.items()
            },
            "timestamp": self.timestamp.isoformat(),
        }


def extract_features_simple(drug1: Dict, drug2: Dict) -> np.ndarray:
    """
    Extract features from drug pair using simple hash encoding.
    Must match the training script's feature extraction.
    """
    n_features = 2 + 100 + 60 + 80  # 242 total
    X = np.zeros(n_features, dtype=np.float32)

    # Boolean features (matched)
    X[0] = 1.0 if drug1.get("matched", drug1.get("drug1_matched", False)) else 0.0
    X[1] = 1.0 if drug2.get("matched", drug2.get("drug2_matched", False)) else 0.0

    def text_hash(text, n_feats, offset):
        if text:
            s = str(text).lower()[:20]
            for j, char in enumerate(s):
                X[offset + hash(char + str(j)) % n_feats] += 1

    col_offset = 2
    text_hash(drug1.get("drug_class", drug1.get("class", "")), 50, col_offset)
    col_offset += 50
    text_hash(drug2.get("drug_class", drug2.get("class", "")), 50, col_offset)
    col_offset += 50
    text_hash(drug1.get("mechanism", ""), 30, col_offset)
    col_offset += 30
    text_hash(drug2.get("mechanism", ""), 30, col_offset)
    col_offset += 30
    text_hash(drug1.get("name", ""), 40, col_offset)
    col_offset += 40
    text_hash(drug2.get("name", ""), 40, col_offset)

    return X


class DDIPredictor:
    """
    Prediction service for Drug-Drug Interactions.

    Uses trained ML models to predict interactions for drug pairs.
    Supports both DDIModel format and direct sklearn models.
    """

    # Severity thresholds based on probability
    SEVERITY_THRESHOLDS = {
        "contraindicated": 0.9,
        "major": 0.7,
        "moderate": 0.4,
        "minor": 0.2,
    }

    MODEL_NAMES = ["random_forest", "xgboost", "lightgbm"]

    # Default threshold - can be overridden by optimal_threshold.json
    DEFAULT_THRESHOLD = 0.5

    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        self.models: Dict[str, any] = {}
        self.is_loaded = False
        self.model_info = {}
        self.use_simple_features = False
        self.optimal_threshold = self.DEFAULT_THRESHOLD
        self.threshold_method = "default"

    def load(self) -> bool:
        """
        Load trained models.

        Returns:
            True if models loaded successfully
        """
        try:
            loaded_count = 0

            for model_name in self.MODEL_NAMES:
                model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")

                if not os.path.exists(model_path):
                    logger.warning(f"Model not found: {model_path}")
                    continue

                try:
                    data = joblib.load(model_path)

                    # Check if it's DDIModel format (dict) or direct sklearn model
                    if isinstance(data, dict) and "model" in data:
                        # DDIModel format
                        model = data.get("calibrated_model") or data.get("model")
                        self.models[model_name] = {
                            "model": model,
                            "is_calibrated": data.get("is_calibrated", False),
                            "metrics": data.get("metrics", {}),
                            "params": data.get("params", {}),
                        }
                    else:
                        # Direct sklearn model (CalibratedClassifierCV or similar)
                        self.models[model_name] = {
                            "model": data,
                            "is_calibrated": True,  # Assume calibrated
                            "metrics": {},
                            "params": {},
                        }
                        self.use_simple_features = True

                    loaded_count += 1
                    logger.info(f"Loaded {model_name} model")

                except Exception as e:
                    logger.error(f"Error loading {model_name}: {e}")

            if loaded_count == 0:
                logger.error("No models found")
                return False

            # Load training results for model info
            results_path = os.path.join(self.model_dir, "training_results.json")
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    self.model_info = json.load(f)

                # Update model metrics from training results
                for model_name in self.models:
                    if model_name in self.model_info.get("models", {}):
                        metrics = self.model_info["models"][model_name].get(
                            "metrics", {}
                        )
                        self.models[model_name]["metrics"] = metrics

            # Load optimal threshold if available
            threshold_path = os.path.join(self.model_dir, "optimal_threshold.json")
            if os.path.exists(threshold_path):
                with open(threshold_path, "r") as f:
                    threshold_data = json.load(f)
                    self.optimal_threshold = threshold_data.get(
                        "threshold", self.DEFAULT_THRESHOLD
                    )
                    self.threshold_method = threshold_data.get("method", "unknown")
                    logger.info(
                        f"Loaded optimal threshold: {self.optimal_threshold:.4f} (method: {self.threshold_method})"
                    )
            else:
                logger.warning(
                    f"No optimal threshold found, using default: {self.DEFAULT_THRESHOLD}"
                )

            self.is_loaded = True
            logger.info(
                f"Predictor loaded with {loaded_count} models (simple_features={self.use_simple_features}, threshold={self.optimal_threshold:.4f})"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading predictor: {e}")
            return False

    def _extract_features(self, drug1: Dict, drug2: Dict) -> np.ndarray:
        """Extract features from drug pair."""
        return extract_features_simple(drug1, drug2)

    def predict(
        self, drug1: Dict, drug2: Dict, threshold: float = None
    ) -> PredictionResult:
        """
        Predict interaction for a drug pair.

        Args:
            drug1: First drug dictionary with properties
            drug2: Second drug dictionary with properties
            threshold: Classification threshold (uses optimal_threshold if None)

        Returns:
            PredictionResult with prediction details
        """
        if not self.is_loaded:
            raise RuntimeError("Predictor not loaded. Call load() first.")

        # Use optimal threshold if not specified
        if threshold is None:
            threshold = self.optimal_threshold

        # Extract features
        features = self._extract_features(drug1, drug2)
        X = features.reshape(1, -1)

        # Get predictions from each model
        model_predictions = {}
        for model_name, model_data in self.models.items():
            try:
                model = model_data["model"]
                proba = model.predict_proba(X)[0, 1]
                model_predictions[model_name] = float(proba)
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")

        if not model_predictions:
            raise RuntimeError("All model predictions failed")

        # Calculate ensemble (average) probability
        ensemble_proba = np.mean(list(model_predictions.values()))
        predicted_interaction = ensemble_proba >= threshold

        # Calculate confidence based on model agreement
        probas = list(model_predictions.values())
        confidence = 1.0 - np.std(probas) if len(probas) > 1 else 0.8

        # Determine severity based on probability
        severity = self._predict_severity(ensemble_proba)

        return PredictionResult(
            drug1_name=drug1.get("name", "Unknown"),
            drug2_name=drug2.get("name", "Unknown"),
            interaction_probability=float(ensemble_proba),
            predicted_interaction=bool(predicted_interaction),
            severity_prediction=severity,
            confidence=float(confidence),
            model_predictions=model_predictions,
        )

    def _predict_severity(self, probability: float) -> str:
        """Predict severity level based on probability."""
        for severity, thresh in self.SEVERITY_THRESHOLDS.items():
            if probability >= thresh:
                return severity
        return "none"

    def predict_batch(
        self, drug_pairs: List[Tuple[Dict, Dict]], threshold: float = None
    ) -> List[PredictionResult]:
        """
        Predict interactions for multiple drug pairs.

        Args:
            drug_pairs: List of (drug1, drug2) tuples
            threshold: Classification threshold

        Returns:
            List of PredictionResult objects
        """
        results = []
        for drug1, drug2 in drug_pairs:
            result = self.predict(drug1, drug2, threshold)
            results.append(result)
        return results

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        if not self.is_loaded:
            return {"error": "Models not loaded"}

        info = {
            "models_loaded": list(self.models.keys()),
            "use_simple_features": self.use_simple_features,
            "optimal_threshold": self.optimal_threshold,
            "threshold_method": self.threshold_method,
            "model_metrics": {},
            "training_info": self.model_info,
        }

        for model_name, model_data in self.models.items():
            info["model_metrics"][model_name] = model_data.get("metrics", {})

        return info

    def get_feature_importance(self) -> Dict[str, List[float]]:
        """Get feature importance from models that support it."""
        importance = {}

        for model_name, model_data in self.models.items():
            model = model_data["model"]

            # Try to get feature importance
            try:
                if hasattr(model, "feature_importances_"):
                    importance[model_name] = model.feature_importances_.tolist()
                elif hasattr(model, "estimator") and hasattr(
                    model.estimator, "feature_importances_"
                ):
                    importance[model_name] = (
                        model.estimator.feature_importances_.tolist()
                    )
            except:
                pass

        return importance


# Global predictor instance
_predictor: Optional[DDIPredictor] = None


def get_predictor(model_dir: str = "./models") -> DDIPredictor:
    """Get or create the global predictor instance."""
    global _predictor

    if _predictor is None:
        _predictor = DDIPredictor(model_dir)
        _predictor.load()

    return _predictor


def predict_interaction(
    drug1: Dict, drug2: Dict, model_dir: str = "./models"
) -> PredictionResult:
    """
    Convenience function to predict interaction.

    Args:
        drug1: First drug dictionary
        drug2: Second drug dictionary
        model_dir: Directory containing trained models

    Returns:
        PredictionResult
    """
    predictor = get_predictor(model_dir)
    return predictor.predict(drug1, drug2)
