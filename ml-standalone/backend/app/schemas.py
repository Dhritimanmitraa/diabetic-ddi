"""
Pydantic schemas for request/response models
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DrugPredictionRequest(BaseModel):
    """Request model for drug interaction prediction"""

    drug1: str = Field(..., min_length=1, description="First drug name")
    drug2: str = Field(..., min_length=1, description="Second drug name")
    include_context: bool = Field(
        default=True, description="Include TWOSIDES database context"
    )
    use_cache: bool = Field(
        default=True, description="Use cached predictions for consistency"
    )


class InteractionContext(BaseModel):
    """TWOSIDES database context for a drug pair"""

    known_interaction: bool
    side_effects: List[str]
    interaction_count: int


class MLPrediction(BaseModel):
    """ML model prediction result"""

    has_interaction: bool
    severity: str  # none, mild, moderate, severe, contraindicated
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None


class ValidationResult(BaseModel):
    """Validation against ground truth"""

    is_validated: bool = Field(description="Whether ground truth exists for this pair")
    is_correct: Optional[bool] = Field(
        default=None, description="Whether prediction matches ground truth"
    )
    ground_truth_severity: Optional[str] = Field(
        default=None, description="Known severity from database"
    )
    calibrated_confidence: float = Field(
        description="Confidence adjusted based on validation"
    )
    message: str = Field(description="Validation status message")


class DrugPrediction(BaseModel):
    """LLM prediction result"""

    has_interaction: bool
    severity: str  # none, mild, moderate, severe, contraindicated
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    mechanism: str
    recommendations: List[str]
    reasoning: str


class DrugPredictionResponse(BaseModel):
    """Response model for drug interaction prediction"""

    drug1: str
    drug2: str
    prediction: DrugPrediction
    ml_prediction: Optional[MLPrediction] = None
    validation: Optional[ValidationResult] = None
    prediction_source: str = Field(
        default="llm", description="Source: 'ml', 'llm', or 'hybrid'"
    )
    is_cached: bool = Field(
        default=False, description="Whether this prediction was from cache"
    )
    twosides_context: Optional[InteractionContext] = None
    llm_model: Optional[str] = None
    processing_time_ms: float


class DrugSearchResponse(BaseModel):
    """Response model for drug search"""

    query: str
    results: List[str]
    total_count: int


class ModelInfo(BaseModel):
    """ML model metadata"""

    is_loaded: bool
    version: Optional[str] = None
    accuracy: Optional[float] = None
    training_date: Optional[str] = None
    num_training_samples: Optional[int] = None
    model_type: Optional[str] = None


class ValidationStats(BaseModel):
    """Validation service statistics"""

    total_validated: int = 0
    correct_predictions: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    accuracy: float = 0.0
    ground_truth_count: int = 0


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    ollama_connected: bool
    database_connected: bool
    ml_model_loaded: bool = False
    ml_model_info: Optional[ModelInfo] = None
    validation_enabled: bool = False
    validation_stats: Optional[ValidationStats] = None
    cached_predictions: int = 0
    model_loaded: Optional[str] = None
    available_models: List[str] = Field(default_factory=list)
    drug_count: int = 0


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
