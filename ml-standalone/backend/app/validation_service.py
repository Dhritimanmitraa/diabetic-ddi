"""
Validation Service for Drug Interaction Predictions

Provides ground truth comparison, confidence calibration, and prediction caching
to ensure reliable and consistent predictions.
"""
import logging
import json
import hashlib
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "cache"


class PredictionCache:
    """Cache for consistent predictions"""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_file = CACHE_DIR / "prediction_cache.json"
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached predictions")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _get_key(self, drug1: str, drug2: str) -> str:
        """Generate consistent cache key for drug pair"""
        # Sort alphabetically for consistency
        drugs = sorted([drug1.lower().strip(), drug2.lower().strip()])
        return hashlib.md5(f"{drugs[0]}:{drugs[1]}".encode()).hexdigest()
    
    def get(self, drug1: str, drug2: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction"""
        key = self._get_key(drug1, drug2)
        return self.cache.get(key)
    
    def set(self, drug1: str, drug2: str, prediction: Dict[str, Any]):
        """Cache a prediction"""
        key = self._get_key(drug1, drug2)
        self.cache[key] = {
            "prediction": prediction,
            "timestamp": datetime.now().isoformat(),
            "drugs": sorted([drug1.lower().strip(), drug2.lower().strip()])
        }
        self._save_cache()
    
    def clear(self):
        """Clear all cached predictions"""
        self.cache = {}
        self._save_cache()


class ValidationService:
    """
    Validates predictions against ground truth and calibrates confidence
    """
    
    def __init__(self):
        self.ground_truth: Dict[str, Dict[str, Any]] = {}
        self.validation_stats = {
            "total_validated": 0,
            "correct_predictions": 0,
            "false_positives": 0,
            "false_negatives": 0
        }
    
    async def load_ground_truth(self, data_service) -> int:
        """Load ground truth from TWOSIDES database"""
        if not data_service or not data_service.is_connected:
            logger.warning("Cannot load ground truth: database not connected")
            return 0
        
        try:
            from sqlalchemy import text
            
            async with data_service.async_session() as session:
                # Load known interactions
                query = text("""
                    SELECT drug1_name, drug2_name, severity, effect
                    FROM twosides_interactions
                    WHERE drug1_name IS NOT NULL AND drug2_name IS NOT NULL
                    LIMIT 100000
                """)
                result = await session.execute(query)
                rows = result.fetchall()
                
                for drug1, drug2, severity, effect in rows:
                    if not drug1 or not drug2:
                        continue
                    
                    drugs = sorted([drug1.lower().strip(), drug2.lower().strip()])
                    key = f"{drugs[0]}:{drugs[1]}"
                    
                    if key not in self.ground_truth:
                        self.ground_truth[key] = {
                            "has_interaction": True,
                            "severity": self._normalize_severity(severity),
                            "effects": []
                        }
                    
                    if effect:
                        self.ground_truth[key]["effects"].append(effect)
                
                logger.info(f"Loaded {len(self.ground_truth)} ground truth interactions")
                return len(self.ground_truth)
                
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            return 0
    
    def _normalize_severity(self, severity) -> str:
        """Normalize severity to standard categories"""
        if severity is None:
            return "mild"
        
        severity_str = str(severity).lower().strip()
        
        mapping = {
            "low": "mild",
            "minor": "mild",
            "medium": "moderate",
            "high": "severe",
            "major": "severe",
            "critical": "contraindicated"
        }
        
        return mapping.get(severity_str, severity_str)
    
    def get_ground_truth(self, drug1: str, drug2: str) -> Optional[Dict[str, Any]]:
        """Get ground truth for a drug pair"""
        drugs = sorted([drug1.lower().strip(), drug2.lower().strip()])
        key = f"{drugs[0]}:{drugs[1]}"
        return self.ground_truth.get(key)
    
    def validate_prediction(
        self, 
        drug1: str, 
        drug2: str, 
        predicted_has_interaction: bool,
        predicted_severity: str
    ) -> Dict[str, Any]:
        """
        Validate a prediction against ground truth
        
        Returns validation result with:
        - is_validated: whether ground truth exists
        - is_correct: whether prediction matches ground truth
        - ground_truth: the actual known interaction (if exists)
        - validation_confidence: adjusted confidence based on validation
        """
        ground_truth = self.get_ground_truth(drug1, drug2)
        
        if ground_truth is None:
            # No ground truth available - prediction is unvalidated
            return {
                "is_validated": False,
                "is_correct": None,
                "ground_truth": None,
                "validation_confidence": 0.5,  # Neutral confidence
                "message": "No ground truth available for this drug pair"
            }
        
        # Compare prediction with ground truth
        self.validation_stats["total_validated"] += 1
        
        # Check if interaction detection is correct
        gt_has_interaction = ground_truth["has_interaction"]
        is_correct = predicted_has_interaction == gt_has_interaction
        
        if is_correct:
            self.validation_stats["correct_predictions"] += 1
            validation_confidence = 1.0
        else:
            if predicted_has_interaction and not gt_has_interaction:
                self.validation_stats["false_positives"] += 1
            else:
                self.validation_stats["false_negatives"] += 1
            validation_confidence = 0.0
        
        # Check severity match (if interaction exists)
        severity_match = None
        if predicted_has_interaction and gt_has_interaction:
            severity_match = predicted_severity == ground_truth["severity"]
        
        return {
            "is_validated": True,
            "is_correct": is_correct,
            "severity_match": severity_match,
            "ground_truth": ground_truth,
            "validation_confidence": validation_confidence,
            "message": "Validated against TWOSIDES database"
        }
    
    def calibrate_confidence(
        self, 
        ml_confidence: float, 
        validation_result: Dict[str, Any]
    ) -> float:
        """
        Calibrate confidence based on validation
        
        Returns adjusted confidence:
        - If validated and correct: boost confidence
        - If validated and incorrect: reduce confidence
        - If not validated: use ML confidence with slight reduction
        """
        if not validation_result["is_validated"]:
            # Unvalidated predictions get slight confidence reduction
            return ml_confidence * 0.85
        
        if validation_result["is_correct"]:
            # Validated correct predictions get confidence boost
            return min(1.0, ml_confidence * 1.1 + 0.1)
        else:
            # Validated incorrect predictions get major confidence reduction
            return ml_confidence * 0.3
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self.validation_stats["total_validated"]
        if total == 0:
            accuracy = 0.0
        else:
            accuracy = self.validation_stats["correct_predictions"] / total
        
        return {
            **self.validation_stats,
            "accuracy": accuracy,
            "ground_truth_count": len(self.ground_truth)
        }


# Global instances
_prediction_cache: Optional[PredictionCache] = None
_validation_service: Optional[ValidationService] = None


def get_prediction_cache() -> PredictionCache:
    """Get the global prediction cache"""
    global _prediction_cache
    if _prediction_cache is None:
        _prediction_cache = PredictionCache()
    return _prediction_cache


def get_validation_service() -> ValidationService:
    """Get the global validation service"""
    global _validation_service
    if _validation_service is None:
        _validation_service = ValidationService()
    return _validation_service


async def initialize_validation_service(data_service) -> ValidationService:
    """Initialize and return the validation service with ground truth"""
    global _validation_service
    _validation_service = ValidationService()
    await _validation_service.load_ground_truth(data_service)
    return _validation_service
