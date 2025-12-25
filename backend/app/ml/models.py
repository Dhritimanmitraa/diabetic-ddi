"""
Machine Learning Models for Drug-Drug Interaction Prediction.

Implements three model types as per the project proposal:
- Random Forest
- XGBoost
- LightGBM

With calibration support (Platt scaling / Isotonic regression).
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report,
    brier_score_loss
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import logging

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Supported model types."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class DDIModel:
    """Base class for DDI prediction models."""
    
    def __init__(self, model_type: ModelType, params: Optional[Dict] = None):
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self.calibrated_model = None
        self.is_trained = False
        self.is_calibrated = False
        self.metrics = {}
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model."""
        raise NotImplementedError
    
    def calibrate(
        self, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        method: str = 'isotonic'
    ):
        """
        Calibrate model probabilities using Platt scaling or Isotonic regression.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            method: 'sigmoid' (Platt) or 'isotonic'
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calibration")
        
        logger.info(f"Calibrating {self.model_type.value} with {method} method...")
        
        # Create calibrated classifier
        self.calibrated_model = CalibratedClassifierCV(
            self.model, 
            method=method, 
            cv='prefit'
        )
        self.calibrated_model.fit(X_val, y_val)
        self.is_calibrated = True
        
        logger.info(f"Calibration complete for {self.model_type.value}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        model = self.calibrated_model if self.is_calibrated else self.model
        return model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (calibrated if available)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        model = self.calibrated_model if self.is_calibrated else self.model
        return model.predict_proba(X)
    
    def predict_proba_uncalibrated(self, X: np.ndarray) -> np.ndarray:
        """Predict raw uncalibrated probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'brier_score': brier_score_loss(y_test, y_proba),
            'is_calibrated': self.is_calibrated,
        }
        
        return self.metrics
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def save(self, filepath: str):
        """Save the trained model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'model_type': self.model_type,
            'params': self.params,
            'is_trained': self.is_trained,
            'is_calibrated': self.is_calibrated,
            'metrics': self.metrics,
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DDIModel':
        """Load a saved model."""
        data = joblib.load(filepath)
        
        instance = cls(data['model_type'], data['params'])
        instance.model = data['model']
        instance.calibrated_model = data.get('calibrated_model')
        instance.is_trained = data['is_trained']
        instance.is_calibrated = data.get('is_calibrated', False)
        instance.metrics = data['metrics']
        
        logger.info(f"Model loaded from {filepath}")
        return instance


class RandomForestDDI(DDIModel):
    """Random Forest classifier for DDI prediction."""
    
    # Default hyperparameter search space
    DEFAULT_PARAM_SPACE = {
        'n_estimators': (50, 500),
        'max_depth': (3, 20),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'max_features': ['sqrt', 'log2', None],
    }
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(ModelType.RANDOM_FOREST, params)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
        }
        default_params.update(self.params)
        self.params = default_params
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Random Forest model."""
        logger.info(f"Training Random Forest with params: {self.params}")
        
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info("Random Forest training complete")


class XGBoostDDI(DDIModel):
    """XGBoost classifier for DDI prediction."""
    
    # Default hyperparameter search space
    DEFAULT_PARAM_SPACE = {
        'n_estimators': (50, 500),
        'max_depth': (3, 15),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'reg_alpha': (0.0, 1.0),
        'reg_lambda': (0.0, 1.0),
    }
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(ModelType.XGBOOST, params)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
        }
        default_params.update(self.params)
        self.params = default_params
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train XGBoost model."""
        logger.info(f"Training XGBoost with params: {self.params}")
        
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info("XGBoost training complete")


class LightGBMDDI(DDIModel):
    """LightGBM classifier for DDI prediction."""
    
    # Default hyperparameter search space
    DEFAULT_PARAM_SPACE = {
        'n_estimators': (50, 500),
        'max_depth': (3, 15),
        'learning_rate': (0.01, 0.3),
        'num_leaves': (20, 150),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'reg_alpha': (0.0, 1.0),
        'reg_lambda': (0.0, 1.0),
    }
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(ModelType.LIGHTGBM, params)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
        default_params.update(self.params)
        self.params = default_params
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train LightGBM model."""
        logger.info(f"Training LightGBM with params: {self.params}")
        
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info("LightGBM training complete")


class DDIModelFactory:
    """Factory for creating DDI models."""
    
    MODEL_CLASSES = {
        ModelType.RANDOM_FOREST: RandomForestDDI,
        ModelType.XGBOOST: XGBoostDDI,
        ModelType.LIGHTGBM: LightGBMDDI,
    }
    
    @classmethod
    def create(cls, model_type: ModelType, params: Optional[Dict] = None) -> DDIModel:
        """Create a model instance."""
        if model_type not in cls.MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls.MODEL_CLASSES[model_type]
        return model_class(params)
    
    @classmethod
    def get_param_space(cls, model_type: ModelType) -> Dict:
        """Get the hyperparameter search space for a model type."""
        model_class = cls.MODEL_CLASSES.get(model_type)
        if model_class and hasattr(model_class, 'DEFAULT_PARAM_SPACE'):
            return model_class.DEFAULT_PARAM_SPACE
        return {}


class EnsemblePredictor:
    """Ensemble predictions from multiple models."""
    
    def __init__(self, models: Dict[ModelType, DDIModel]):
        self.models = models
        self.weights = {mt: 1.0 / len(models) for mt in models}
    
    def set_weights(self, weights: Dict[ModelType, float]):
        """Set custom weights for ensemble."""
        total = sum(weights.values())
        self.weights = {mt: w / total for mt, w in weights.items()}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get weighted ensemble probability predictions."""
        probas = []
        weights = []
        
        for model_type, model in self.models.items():
            if model.is_trained:
                proba = model.predict_proba(X)[:, 1]
                probas.append(proba)
                weights.append(self.weights.get(model_type, 1.0))
        
        if not probas:
            raise ValueError("No trained models available")
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_proba = np.average(probas, axis=0, weights=weights)
        
        return ensemble_proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get ensemble class predictions."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

