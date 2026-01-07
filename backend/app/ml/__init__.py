"""
Machine Learning module for Drug-Drug Interaction Prediction.

This module implements:
- Feature engineering for drug pairs
- Multiple ML models (Random Forest, XGBoost, LightGBM)
- Bayesian hyperparameter optimization using Optuna
- Model training and prediction pipelines
"""

from app.ml.feature_engineering import DrugFeatureExtractor
from app.ml.models import DDIModelFactory, ModelType
from app.ml.bayesian_optimizer import BayesianOptimizer, OptimizationMethod
from app.ml.trainer import DDITrainer
from app.ml.predictor import DDIPredictor

__all__ = [
    "DrugFeatureExtractor",
    "DDIModelFactory",
    "ModelType",
    "BayesianOptimizer",
    "OptimizationMethod",
    "DDITrainer",
    "DDIPredictor",
]
