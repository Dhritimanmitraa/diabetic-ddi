"""
ML Model Factory for Drug-Drug Interaction Prediction.

Provides factory pattern for creating various ML models.
"""

from enum import Enum
from typing import Any, Dict, Optional
import logging
import os
import numpy as np
import joblib

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Supported model types."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING = "gradient_boosting"


class DDIModelFactory:
    """Factory for creating DDI prediction models."""
    
    @staticmethod
    def create_model(model_type: ModelType, params: Optional[Dict[str, Any]] = None):
        """
        Create a raw sklearn-compatible model instance.
        
        Args:
            model_type: Type of model to create
            params: Model hyperparameters
            
        Returns:
            Configured sklearn-compatible model instance
        """
        params = params or {}
        
        if model_type == ModelType.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestClassifier
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }
            default_params.update(params)
            return RandomForestClassifier(**default_params)
        
        elif model_type == ModelType.XGBOOST:
            try:
                from xgboost import XGBClassifier
                default_params = {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "n_jobs": -1,
                    "use_label_encoder": False,
                    "eval_metric": "mlogloss",
                }
                default_params.update(params)
                return XGBClassifier(**default_params)
            except ImportError:
                logger.warning("XGBoost not installed, falling back to GradientBoosting")
                return DDIModelFactory.create_model(ModelType.GRADIENT_BOOSTING, params)
        
        elif model_type == ModelType.LIGHTGBM:
            try:
                from lightgbm import LGBMClassifier
                default_params = {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "num_leaves": 31,
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1,
                }
                default_params.update(params)
                return LGBMClassifier(**default_params)
            except ImportError:
                logger.warning("LightGBM not installed, falling back to GradientBoosting")
                return DDIModelFactory.create_model(ModelType.GRADIENT_BOOSTING, params)
        
        elif model_type == ModelType.LOGISTIC_REGRESSION:
            from sklearn.linear_model import LogisticRegression
            default_params = {
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }
            default_params.update(params)
            return LogisticRegression(**default_params)
        
        elif model_type == ModelType.GRADIENT_BOOSTING:
            from sklearn.ensemble import GradientBoostingClassifier
            default_params = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": 42,
            }
            default_params.update(params)
            return GradientBoostingClassifier(**default_params)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create(model_type: ModelType, params: Optional[Dict[str, Any]] = None) -> "DDIModel":
        """
        Create a DDIModel wrapper with an underlying sklearn model.

        Args:
            model_type: Type of model to create
            params: Model hyperparameters

        Returns:
            DDIModel wrapping the configured sklearn model
        """
        raw_model = DDIModelFactory.create_model(model_type, params)
        return DDIModel(raw_model, model_type, params=params)

    @staticmethod
    def get_param_space(model_type: ModelType) -> Dict[str, Any]:
        """Get hyperparameter search space for a model type."""
        if model_type == ModelType.RANDOM_FOREST:
            return {
                "n_estimators": (50, 300),
                "max_depth": (3, 20),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10),
            }
        elif model_type == ModelType.XGBOOST:
            return {
                "n_estimators": (50, 300),
                "max_depth": (3, 12),
                "learning_rate": (0.01, 0.3),
                "subsample": (0.6, 1.0),
                "colsample_bytree": (0.6, 1.0),
            }
        elif model_type == ModelType.LIGHTGBM:
            return {
                "n_estimators": (50, 300),
                "max_depth": (3, 12),
                "learning_rate": (0.01, 0.3),
                "num_leaves": (10, 100),
            }
        else:
            return {}


class DDIModel:
    """
    Wrapper class for trained DDI models.

    Provides a unified interface (train / evaluate / save / load) around
    different sklearn-compatible estimators so the training pipeline can
    treat all model types identically.
    """

    def __init__(
        self,
        model: Any,
        model_type: ModelType,
        feature_extractor: Any = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.model_type = model_type
        self.feature_extractor = feature_extractor
        self.params: Dict[str, Any] = params or {}
        self.metrics: Dict[str, float] = {}
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Training & Evaluation
    # ------------------------------------------------------------------

    def train(self, X, y, **kwargs):
        """Train the underlying model."""
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    # keep sklearn-style alias
    def fit(self, X, y, **kwargs):
        return self.train(X, y, **kwargs)

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate performance on a test set and store metrics.

        Returns:
            Dictionary of metric_name -> value.
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score,
        )

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_test, y_proba)),
        }
        return self.metrics

    def get_params(self):
        """Get model parameters."""
        return self.model.get_params()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        """Save model, params, and metrics to disk."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "model": self.model,
            "model_type": self.model_type.value,
            "params": self.params,
            "metrics": self.metrics,
            "is_calibrated": False,
        }
        joblib.dump(data, filepath)
        logger.info(f"Saved {self.model_type.value} model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "DDIModel":
        """Load a previously-saved DDIModel."""
        data = joblib.load(filepath)
        if isinstance(data, dict) and "model" in data:
            model_type = ModelType(data.get("model_type", "random_forest"))
            instance = cls(
                model=data["model"],
                model_type=model_type,
                params=data.get("params", {}),
            )
            instance.metrics = data.get("metrics", {})
            instance.is_fitted = True
            return instance
        # Legacy: raw sklearn model stored directly
        instance = cls(model=data, model_type=ModelType.RANDOM_FOREST)
        instance.is_fitted = True
        return instance


class EnsemblePredictor:
    """Ensemble of multiple DDI models."""
    
    def __init__(self, models: list):
        self.models = models if isinstance(models, list) else list(models.values())
    
    def predict(self, X):
        """Make ensemble predictions using voting."""
        predictions = [m.predict(X) for m in self.models]
        stacked = np.stack(predictions, axis=0)
        from scipy import stats
        modes, _ = stats.mode(stacked, axis=0, keepdims=False)
        return modes
    
    def predict_proba(self, X):
        """Average probability predictions."""
        probas = [m.predict_proba(X) for m in self.models]
        return np.mean(probas, axis=0)
