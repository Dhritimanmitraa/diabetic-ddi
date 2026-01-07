"""
ML Model Factory for Drug-Drug Interaction Prediction.

Provides factory pattern for creating various ML models.
"""

from enum import Enum
from typing import Any, Dict, Optional
import logging

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
        Create a model instance.

        Args:
            model_type: Type of model to create
            params: Model hyperparameters

        Returns:
            Configured model instance
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
                logger.warning(
                    "XGBoost not installed, falling back to GradientBoosting"
                )
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
                logger.warning(
                    "LightGBM not installed, falling back to GradientBoosting"
                )
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
    """Wrapper class for trained DDI models."""

    def __init__(
        self, model: Any, model_type: ModelType, feature_extractor: Any = None
    ):
        self.model = model
        self.model_type = model_type
        self.feature_extractor = feature_extractor
        self.is_fitted = False

    def fit(self, X, y, **kwargs):
        """Train the model."""
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.model.predict_proba(X)

    def get_params(self):
        """Get model parameters."""
        return self.model.get_params()


class EnsemblePredictor:
    """Ensemble of multiple DDI models."""

    def __init__(self, models: list):
        self.models = models

    def predict(self, X):
        """Make ensemble predictions using voting."""
        import numpy as np

        predictions = [m.predict(X) for m in self.models]
        # Majority voting
        stacked = np.stack(predictions, axis=0)
        from scipy import stats

        modes, _ = stats.mode(stacked, axis=0, keepdims=False)
        return modes

    def predict_proba(self, X):
        """Average probability predictions."""
        import numpy as np

        probas = [m.predict_proba(X) for m in self.models]
        return np.mean(probas, axis=0)
