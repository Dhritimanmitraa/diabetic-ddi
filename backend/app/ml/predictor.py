"""
Prediction Service for Drug-Drug Interaction.

Provides inference using trained ML models.
Compatible with both DDIModel format and direct sklearn models.

Example usage:
    >>> from app.ml.predictor import predict_interaction
    >>> drug1 = {'name': 'Aspirin', 'drug_class': 'NSAID', 'mechanism': 'COX inhibitor'}
    >>> drug2 = {'name': 'Warfarin', 'drug_class': 'Anticoagulant', 'mechanism': 'VKA'}
    >>> result = predict_interaction(drug1, drug2)
    >>> print(f"Interaction probability: {result.interaction_probability:.2%}")
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import joblib
import numpy as np
import numpy.typing as npt

from app.ml.feature_engineering import DrugFeatureExtractor

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

class DrugDict(TypedDict, total=False):
    """Type definition for drug information dictionary."""
    name: str
    generic_name: Optional[str]
    drug_class: Optional[str]
    description: Optional[str]
    mechanism: Optional[str]
    indication: Optional[str]
    molecular_weight: Optional[float]
    is_approved: bool
    matched: bool
    drug1_matched: bool  # Alternative key for matched
    drug2_matched: bool  # Alternative key for matched


class ModelMetricsDict(TypedDict, total=False):
    """Type definition for model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    npv: float
    specificity: float


class ModelDataDict(TypedDict):
    """Type definition for loaded model data."""
    model: Any  # sklearn-compatible model with predict_proba method
    is_calibrated: bool
    metrics: ModelMetricsDict
    params: Dict[str, Any]


class TrainingInfoDict(TypedDict, total=False):
    """Type definition for training information."""
    models: Dict[str, Dict[str, Any]]
    training_date: str
    dataset_size: int
    feature_count: int


class ThresholdDataDict(TypedDict):
    """Type definition for optimal threshold configuration."""
    threshold: float
    method: str


class ModelInfoDict(TypedDict):
    """Type definition for model information response."""
    models_loaded: List[str]
    use_simple_features: bool
    optimal_threshold: float
    threshold_method: str
    model_metrics: Dict[str, ModelMetricsDict]
    training_info: TrainingInfoDict


class PredictionResultDict(TypedDict):
    """Type definition for prediction result dictionary."""
    drug1: str
    drug2: str
    interaction_probability: float
    predicted_interaction: bool
    severity_prediction: str
    confidence: float
    model_predictions: Dict[str, float]
    timestamp: str


# Type alias for feature array
FeatureArray = npt.NDArray[np.float32]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PredictionResult:
    """
    Container for prediction results.
    
    Attributes:
        drug1_name: Name of the first drug
        drug2_name: Name of the second drug
        interaction_probability: Ensemble probability of interaction (0.0 to 1.0)
        predicted_interaction: Binary prediction based on threshold
        severity_prediction: Predicted severity level
        confidence: Model agreement confidence (0.0 to 1.0)
        model_predictions: Individual model probability predictions
        timestamp: When the prediction was made
        
    Example:
        >>> result = PredictionResult(
        ...     drug1_name="Aspirin",
        ...     drug2_name="Warfarin",
        ...     interaction_probability=0.85,
        ...     predicted_interaction=True,
        ...     severity_prediction="major",
        ...     confidence=0.92,
        ...     model_predictions={"random_forest": 0.87, "xgboost": 0.83}
        ... )
        >>> result.to_dict()
    """
    drug1_name: str
    drug2_name: str
    interaction_probability: float
    predicted_interaction: bool
    severity_prediction: str
    confidence: float
    model_predictions: Dict[str, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> PredictionResultDict:
        """Convert to dictionary for JSON serialization."""
        return {
            'drug1': self.drug1_name,
            'drug2': self.drug2_name,
            'interaction_probability': round(self.interaction_probability, 4),
            'predicted_interaction': self.predicted_interaction,
            'severity_prediction': self.severity_prediction,
            'confidence': round(self.confidence, 4),
            'model_predictions': {
                k: round(v, 4) for k, v in self.model_predictions.items()
            },
            'timestamp': self.timestamp.isoformat(),
        }


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_features_simple(drug1: DrugDict, drug2: DrugDict) -> FeatureArray:
    """
    Extract features from drug pair using simple hash encoding.
    
    This must match the training script's feature extraction to ensure
    model compatibility.
    
    Args:
        drug1: First drug dictionary with properties
        drug2: Second drug dictionary with properties
        
    Returns:
        numpy array of shape (242,) with float32 dtype
        
    Feature structure:
        - [0-1]: Boolean matched features (2)
        - [2-51]: Drug 1 class hash features (50)
        - [52-101]: Drug 2 class hash features (50)
        - [102-131]: Drug 1 mechanism hash features (30)
        - [132-161]: Drug 2 mechanism hash features (30)
        - [162-201]: Drug 1 name hash features (40)
        - [202-241]: Drug 2 name hash features (40)
    """
    n_features: int = 2 + 100 + 60 + 80  # 242 total
    X: FeatureArray = np.zeros(n_features, dtype=np.float32)
    
    # Boolean features (matched)
    X[0] = 1.0 if drug1.get('matched', drug1.get('drug1_matched', False)) else 0.0
    X[1] = 1.0 if drug2.get('matched', drug2.get('drug2_matched', False)) else 0.0
    
    def text_hash(text: Optional[str], n_feats: int, offset: int) -> None:
        """Hash text characters into feature buckets."""
        if text:
            s = str(text).lower()[:20]
            for j, char in enumerate(s):
                X[offset + hash(char + str(j)) % n_feats] += 1
    
    col_offset: int = 2
    text_hash(drug1.get('drug_class', drug1.get('class', '')), 50, col_offset)
    col_offset += 50
    text_hash(drug2.get('drug_class', drug2.get('class', '')), 50, col_offset)
    col_offset += 50
    text_hash(drug1.get('mechanism', ''), 30, col_offset)
    col_offset += 30
    text_hash(drug2.get('mechanism', ''), 30, col_offset)
    col_offset += 30
    text_hash(drug1.get('name', ''), 40, col_offset)
    col_offset += 40
    text_hash(drug2.get('name', ''), 40, col_offset)
    
    return X


# =============================================================================
# Predictor Class
# =============================================================================

class DDIPredictor:
    """
    Prediction service for Drug-Drug Interactions.
    
    Uses trained ML models to predict interactions for drug pairs.
    Supports both DDIModel format and direct sklearn models.
    
    Attributes:
        model_dir: Directory containing trained model files
        models: Dictionary of loaded model data
        is_loaded: Whether models have been successfully loaded
        optimal_threshold: Classification threshold for binary prediction
        
    Example:
        >>> predictor = DDIPredictor("./models")
        >>> predictor.load()
        True
        >>> result = predictor.predict(drug1_dict, drug2_dict)
        >>> print(result.severity_prediction)
        'major'
    """
    
    # Severity thresholds based on probability
    SEVERITY_THRESHOLDS: Dict[str, float] = {
        'contraindicated': 0.9,
        'major': 0.7,
        'moderate': 0.4,
        'minor': 0.2,
    }
    
    MODEL_NAMES: List[str] = ['random_forest', 'xgboost', 'lightgbm']
    
    # Default threshold - can be overridden by optimal_threshold.json
    DEFAULT_THRESHOLD: float = 0.5
    
    def __init__(self, model_dir: str = "./models") -> None:
        """
        Initialize the predictor.
        
        Args:
            model_dir: Directory containing trained model files
        """
        self.model_dir: str = model_dir
        self.models: Dict[str, ModelDataDict] = {}
        self.is_loaded: bool = False
        self.model_info: TrainingInfoDict = {}
        self.use_simple_features: bool = False
        self.optimal_threshold: float = self.DEFAULT_THRESHOLD
        self.threshold_method: str = "default"
        self.feature_extractor: Optional[DrugFeatureExtractor] = None
        
    def load(self) -> bool:
        """
        Load trained models from disk.
        
        Returns:
            True if at least one model was loaded successfully, False otherwise
            
        Raises:
            No exceptions are raised; errors are logged and False is returned
        """
        try:
            loaded_count: int = 0
            
            for model_name in self.MODEL_NAMES:
                model_path: str = os.path.join(self.model_dir, f"{model_name}_model.pkl")
                
                if not os.path.exists(model_path):
                    logger.warning(f"Model not found: {model_path}")
                    continue
                
                try:
                    data: Any = joblib.load(model_path)
                    
                    # Check if it's DDIModel format (dict) or direct sklearn model
                    if isinstance(data, dict) and 'model' in data:
                        # DDIModel format
                        model = data.get('calibrated_model') or data.get('model')
                        self.models[model_name] = {
                            'model': model,
                            'is_calibrated': data.get('is_calibrated', False),
                            'metrics': data.get('metrics', {}),
                            'params': data.get('params', {})
                        }
                    else:
                        # Direct sklearn model (CalibratedClassifierCV or similar)
                        self.models[model_name] = {
                            'model': data,
                            'is_calibrated': True,  # Assume calibrated
                            'metrics': {},
                            'params': {}
                        }
                        self.use_simple_features = True
                    
                    loaded_count += 1
                    logger.info(f"Loaded {model_name} model")
                    
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {e}")
            
            if loaded_count == 0:
                logger.error("No models found")
                return False
            
            # Load trained feature extractor if available
            self._load_feature_extractor()
            
            # Load training results for model info
            self._load_training_results()
            
            # Load optimal threshold if available
            self._load_optimal_threshold()
            
            self.is_loaded = True
            logger.info(
                f"Predictor loaded with {loaded_count} models "
                f"(simple_features={self.use_simple_features}, "
                f"threshold={self.optimal_threshold:.4f})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error loading predictor: {e}")
            return False
    
    def _load_training_results(self) -> None:
        """Load training results JSON file if available."""
        results_path: str = os.path.join(self.model_dir, "training_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                self.model_info = json.load(f)
                
            # Update model metrics from training results
            for model_name in self.models:
                if model_name in self.model_info.get('models', {}):
                    metrics: ModelMetricsDict = self.model_info['models'][model_name].get('metrics', {})
                    self.models[model_name]['metrics'] = metrics
    
    def _load_feature_extractor(self) -> None:
        """Load the fitted DrugFeatureExtractor if available."""
        fe_path: str = os.path.join(self.model_dir, "feature_extractor.pkl")
        if os.path.exists(fe_path):
            try:
                self.feature_extractor = DrugFeatureExtractor.load(fe_path)
                self.use_simple_features = False
                logger.info("Loaded trained feature extractor — using canonical features")
            except Exception as e:
                logger.warning(f"Failed to load feature extractor, falling back to simple features: {e}")
                self.feature_extractor = None
                self.use_simple_features = True
        else:
            logger.warning("No feature_extractor.pkl found — falling back to simple hash features")
            self.use_simple_features = True

    def _load_optimal_threshold(self) -> None:
        """Load optimal threshold configuration if available."""
        threshold_path: str = os.path.join(self.model_dir, "optimal_threshold.json")
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r', encoding='utf-8') as f:
                threshold_data: ThresholdDataDict = json.load(f)
                self.optimal_threshold = threshold_data.get('threshold', self.DEFAULT_THRESHOLD)
                self.threshold_method = threshold_data.get('method', 'unknown')
                logger.info(
                    f"Loaded optimal threshold: {self.optimal_threshold:.4f} "
                    f"(method: {self.threshold_method})"
                )
        else:
            logger.warning(f"No optimal threshold found, using default: {self.DEFAULT_THRESHOLD}")
    
    def _extract_features(self, drug1: DrugDict, drug2: DrugDict) -> FeatureArray:
        """
        Extract features from drug pair.

        Uses the trained DrugFeatureExtractor (TF-IDF + categorical +
        statistical features) when available, otherwise falls back to
        the legacy 242-dim hash encoder for backward compatibility with
        pre-existing model artifacts.

        Args:
            drug1: First drug dictionary
            drug2: Second drug dictionary
            
        Returns:
            Feature array for model input
        """
        if self.feature_extractor is not None and self.feature_extractor.is_fitted:
            return self.feature_extractor.extract_features(drug1, drug2)
        return extract_features_simple(drug1, drug2)
    
    def predict(
        self,
        drug1: DrugDict,
        drug2: DrugDict,
        threshold: Optional[float] = None
    ) -> PredictionResult:
        """
        Predict interaction for a drug pair.
        
        Args:
            drug1: First drug dictionary with properties (name, drug_class, mechanism, etc.)
            drug2: Second drug dictionary with properties
            threshold: Classification threshold for binary prediction.
                      Uses optimal_threshold if None.
            
        Returns:
            PredictionResult with prediction details including probability,
            binary prediction, severity, and confidence.
            
        Raises:
            RuntimeError: If predictor not loaded or all model predictions fail
            
        Example:
            >>> predictor = DDIPredictor("./models")
            >>> predictor.load()
            >>> result = predictor.predict(
            ...     {"name": "Aspirin", "drug_class": "NSAID"},
            ...     {"name": "Warfarin", "drug_class": "Anticoagulant"}
            ... )
            >>> print(f"Probability: {result.interaction_probability:.2%}")
        """
        if not self.is_loaded:
            raise RuntimeError("Predictor not loaded. Call load() first.")
        
        # Use optimal threshold if not specified
        effective_threshold: float = threshold if threshold is not None else self.optimal_threshold
        
        # Extract features
        features: FeatureArray = self._extract_features(drug1, drug2)
        X: npt.NDArray[np.float32] = features.reshape(1, -1)
        
        # Get predictions from each model
        model_predictions: Dict[str, float] = {}
        for model_name, model_data in self.models.items():
            try:
                model = model_data['model']
                proba: float = float(model.predict_proba(X)[0, 1])
                model_predictions[model_name] = proba
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
        
        if not model_predictions:
            raise RuntimeError("All model predictions failed")
        
        # Calculate ensemble (average) probability
        ensemble_proba: float = float(np.mean(list(model_predictions.values())))
        predicted_interaction: bool = ensemble_proba >= effective_threshold
        
        # Calculate confidence based on model agreement
        probas: List[float] = list(model_predictions.values())
        confidence: float = 1.0 - float(np.std(probas)) if len(probas) > 1 else 0.8
        
        # Determine severity based on probability
        severity: str = self._predict_severity(ensemble_proba)
        
        return PredictionResult(
            drug1_name=drug1.get('name', 'Unknown'),
            drug2_name=drug2.get('name', 'Unknown'),
            interaction_probability=ensemble_proba,
            predicted_interaction=predicted_interaction,
            severity_prediction=severity,
            confidence=confidence,
            model_predictions=model_predictions
        )
    
    def _predict_severity(self, probability: float) -> str:
        """
        Predict severity level based on probability.
        
        Args:
            probability: Interaction probability (0.0 to 1.0)
            
        Returns:
            Severity level string: 'contraindicated', 'major', 'moderate', 
            'minor', or 'none'
        """
        for severity, thresh in self.SEVERITY_THRESHOLDS.items():
            if probability >= thresh:
                return severity
        return 'none'
    
    def predict_batch(
        self,
        drug_pairs: List[Tuple[DrugDict, DrugDict]],
        threshold: Optional[float] = None
    ) -> List[PredictionResult]:
        """
        Predict interactions for multiple drug pairs using vectorized operations.
        
        This method uses batch prediction for better performance, especially
        when checking many drug pairs at once.
        
        Args:
            drug_pairs: List of (drug1, drug2) tuples
            threshold: Classification threshold for all predictions
            
        Returns:
            List of PredictionResult objects in the same order as input
            
        Example:
            >>> pairs = [(drug1, drug2), (drug3, drug4)]
            >>> results = predictor.predict_batch(pairs)
            >>> for result in results:
            ...     print(f"{result.drug1_name} + {result.drug2_name}: {result.severity_prediction}")
        """
        if not drug_pairs:
            return []
        
        # Use vectorized version for better performance
        return self.predict_batch_vectorized(drug_pairs, threshold)
    
    def predict_batch_vectorized(
        self,
        drug_pairs: List[Tuple[DrugDict, DrugDict]],
        threshold: Optional[float] = None
    ) -> List[PredictionResult]:
        """
        Predict interactions for multiple drug pairs using vectorized operations.
        
        This method extracts features for all pairs at once and performs batch
        prediction, which is significantly faster than sequential prediction
        when processing many pairs.
        
        Performance improvement:
            - 10-50x faster than sequential prediction for 100+ pairs
            - Memory efficient using numpy array operations
        
        Args:
            drug_pairs: List of (drug1, drug2) tuples
            threshold: Classification threshold for all predictions
            
        Returns:
            List of PredictionResult objects in the same order as input
            
        Example:
            >>> # Check interactions for all pairs in a patient's medication list
            >>> medications = [drug1, drug2, drug3, drug4]
            >>> pairs = [(m1, m2) for i, m1 in enumerate(medications) 
            ...          for m2 in medications[i+1:]]
            >>> results = predictor.predict_batch_vectorized(pairs)
        """
        if not self.is_loaded:
            raise RuntimeError("Predictor not loaded. Call load() first.")
        
        if not drug_pairs:
            return []
        
        # Use optimal threshold if not specified
        effective_threshold: float = threshold if threshold is not None else self.optimal_threshold
        
        n_pairs = len(drug_pairs)
        
        # Extract all features at once (vectorized)
        X: npt.NDArray[np.float32] = np.vstack([
            self._extract_features(drug1, drug2)
            for drug1, drug2 in drug_pairs
        ])
        
        # Get batch predictions from each model
        model_probas: Dict[str, npt.NDArray[np.float64]] = {}
        
        for model_name, model_data in self.models.items():
            try:
                model = model_data['model']
                # Batch prediction - much faster than individual calls
                probas = model.predict_proba(X)[:, 1]
                model_probas[model_name] = probas
            except Exception as e:
                logger.warning(f"Batch prediction failed for {model_name}: {e}")
        
        if not model_probas:
            raise RuntimeError("All model batch predictions failed")
        
        # Stack all model predictions for ensemble calculation
        all_probas: npt.NDArray[np.float64] = np.vstack(list(model_probas.values()))
        
        # Calculate ensemble (average) probabilities - vectorized
        ensemble_probas: npt.NDArray[np.float64] = np.mean(all_probas, axis=0)
        
        # Calculate confidence based on model agreement - vectorized
        confidences: npt.NDArray[np.float64] = 1.0 - np.std(all_probas, axis=0) if len(model_probas) > 1 else np.full(n_pairs, 0.8)
        
        # Binary predictions based on threshold - vectorized
        predicted_interactions: npt.NDArray[np.bool_] = ensemble_probas >= effective_threshold
        
        # Build results
        results: List[PredictionResult] = []
        for i, (drug1, drug2) in enumerate(drug_pairs):
            # Get individual model predictions for this pair
            individual_preds = {
                model_name: float(probas[i])
                for model_name, probas in model_probas.items()
            }
            
            results.append(PredictionResult(
                drug1_name=drug1.get('name', 'Unknown'),
                drug2_name=drug2.get('name', 'Unknown'),
                interaction_probability=float(ensemble_probas[i]),
                predicted_interaction=bool(predicted_interactions[i]),
                severity_prediction=self._predict_severity(float(ensemble_probas[i])),
                confidence=float(confidences[i]),
                model_predictions=individual_preds
            ))
        
        return results
    
    def get_model_info(self) -> Union[ModelInfoDict, Dict[str, str]]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary containing model metadata, metrics, and configuration.
            Returns error dict if models not loaded.
        """
        if not self.is_loaded:
            return {'error': 'Models not loaded'}
        
        model_metrics: Dict[str, ModelMetricsDict] = {}
        for model_name, model_data in self.models.items():
            model_metrics[model_name] = model_data.get('metrics', {})
        
        return {
            'models_loaded': list(self.models.keys()),
            'use_simple_features': self.use_simple_features,
            'optimal_threshold': self.optimal_threshold,
            'threshold_method': self.threshold_method,
            'model_metrics': model_metrics,
            'training_info': self.model_info,
        }
    
    def get_feature_importance(self) -> Dict[str, List[float]]:
        """
        Get feature importance from models that support it.
        
        Returns:
            Dictionary mapping model names to their feature importance arrays.
            Only includes models that have feature_importances_ attribute.
        """
        importance: Dict[str, List[float]] = {}
        
        for model_name, model_data in self.models.items():
            model = model_data['model']
            
            # Try to get feature importance
            try:
                if hasattr(model, 'feature_importances_'):
                    importance[model_name] = model.feature_importances_.tolist()
                elif hasattr(model, 'estimator') and hasattr(model.estimator, 'feature_importances_'):
                    importance[model_name] = model.estimator.feature_importances_.tolist()
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Feature importance unavailable for {model_name}: {e}")
        
        return importance


# =============================================================================
# Module-level Singleton and Convenience Functions
# =============================================================================

# Global predictor instance for singleton pattern
_predictor: Optional[DDIPredictor] = None


def get_predictor(model_dir: str = "./models") -> DDIPredictor:
    """
    Get or create the global predictor instance.
    
    This function implements a singleton pattern to avoid reloading models
    on each prediction request.
    
    Args:
        model_dir: Directory containing trained models (only used on first call)
        
    Returns:
        Loaded DDIPredictor instance
        
    Note:
        The model_dir parameter is only used on the first call. Subsequent
        calls return the same instance regardless of model_dir value.
    """
    global _predictor
    
    if _predictor is None:
        _predictor = DDIPredictor(model_dir)
        _predictor.load()
    
    return _predictor


def predict_interaction(
    drug1: DrugDict,
    drug2: DrugDict,
    model_dir: str = "./models"
) -> PredictionResult:
    """
    Convenience function to predict interaction.
    
    Uses the global predictor singleton for efficiency.
    
    Args:
        drug1: First drug dictionary with properties
        drug2: Second drug dictionary with properties
        model_dir: Directory containing trained models (only used on first call)
        
    Returns:
        PredictionResult with prediction details
        
    Example:
        >>> from app.ml.predictor import predict_interaction
        >>> result = predict_interaction(
        ...     {"name": "Aspirin", "drug_class": "NSAID"},
        ...     {"name": "Warfarin", "drug_class": "Anticoagulant"}
        ... )
        >>> if result.predicted_interaction:
        ...     print(f"Warning: {result.severity_prediction} interaction!")
    """
    predictor = get_predictor(model_dir)
    return predictor.predict(drug1, drug2)
