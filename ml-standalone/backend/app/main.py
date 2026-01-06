"""
ML Standalone Application - FastAPI Backend
Hybrid ML + LLM drug interaction prediction with validation and caching
"""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .llm_service import get_llm_service, initialize_llm_service
from .ml_model import get_ml_model, initialize_ml_model
from .data_service import DataService
from .validation_service import (
    get_prediction_cache, 
    get_validation_service, 
    initialize_validation_service
)
from .schemas import (
    DrugPredictionRequest,
    DrugPredictionResponse,
    DrugSearchResponse,
    HealthResponse,
    ErrorResponse,
    MLPrediction,
    ModelInfo,
    ValidationResult,
    ValidationStats,
    DrugPrediction
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
llm_service = None
ml_model = None
data_service = None
validation_service = None
prediction_cache = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global llm_service, ml_model, data_service, validation_service, prediction_cache
    
    logger.info("Starting ML Standalone Application...")
    
    # Initialize prediction cache
    prediction_cache = get_prediction_cache()
    logger.info(f"Prediction cache ready with {len(prediction_cache.cache)} cached predictions")
    
    # Initialize ML model first (fast, doesn't require Ollama)
    try:
        ml_model = initialize_ml_model()
        if ml_model.is_loaded:
            logger.info(f"ML Model ready - Version: {ml_model.model_info.get('version', 'N/A')}")
        else:
            logger.warning("ML Model not loaded. Run train_model.py to train the model.")
    except Exception as e:
        logger.error(f"Failed to initialize ML model: {e}")
    
    # Initialize LLM service
    try:
        llm_service = await initialize_llm_service(model="llama3.2")
        if llm_service:
            logger.info(f"LLM Service ready with model: {llm_service.model}")
        else:
            logger.warning("LLM Service initialization failed")
    except Exception as e:
        logger.error(f"Failed to initialize LLM service: {e}")
    
    # Initialize Data service
    try:
        data_service = DataService()
        await data_service.initialize()
        drug_count = await data_service.get_drug_count()
        logger.info(f"Data Service ready with {drug_count} drugs")
    except Exception as e:
        logger.error(f"Failed to initialize Data service: {e}")
    
    # Initialize Validation service with ground truth
    try:
        validation_service = await initialize_validation_service(data_service)
        gt_count = len(validation_service.ground_truth)
        logger.info(f"Validation Service ready with {gt_count} ground truth interactions")
    except Exception as e:
        logger.error(f"Failed to initialize Validation service: {e}")
    
    yield
    
    logger.info("Shutting down ML Standalone Application...")


# Create FastAPI app
app = FastAPI(
    title="ML Standalone Drug Interaction Predictor",
    description="Hybrid ML + LLM drug interaction prediction with validation and caching",
    version="2.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "ML Standalone Drug Interaction Predictor API",
        "version": "2.1.0",
        "features": [
            "ML Model (XGBoost)",
            "LLM Explanations",
            "Ground Truth Validation",
            "Prediction Caching",
            "Confidence Calibration"
        ],
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns the status of ML model, Ollama connection, database, and validation service.
    """
    ollama_connected = False
    model_loaded = None
    available_models = []
    
    if llm_service:
        ollama_connected = await llm_service.check_connection_async()
        if ollama_connected:
            model_loaded = llm_service.model
            available_models = llm_service.available_models
    
    database_connected = data_service.is_connected if data_service else False
    drug_count = 0
    
    if data_service and data_service.is_connected:
        drug_count = await data_service.get_drug_count()
    
    # ML Model info
    ml_model_loaded = ml_model.is_loaded if ml_model else False
    ml_model_info = None
    if ml_model and ml_model.is_loaded:
        info = ml_model.get_model_info()
        ml_model_info = ModelInfo(
            is_loaded=info.get("is_loaded", False),
            version=info.get("version"),
            accuracy=info.get("accuracy"),
            training_date=info.get("training_date"),
            num_training_samples=info.get("num_training_samples"),
            model_type=info.get("model_type")
        )
    
    # Validation stats
    validation_enabled = validation_service is not None and len(validation_service.ground_truth) > 0
    validation_stats_data = None
    if validation_service:
        stats = validation_service.get_validation_stats()
        validation_stats_data = ValidationStats(
            total_validated=stats["total_validated"],
            correct_predictions=stats["correct_predictions"],
            false_positives=stats["false_positives"],
            false_negatives=stats["false_negatives"],
            accuracy=stats["accuracy"],
            ground_truth_count=stats["ground_truth_count"]
        )
    
    # Cache stats
    cached_predictions = len(prediction_cache.cache) if prediction_cache else 0
    
    # Status is healthy if ML model is loaded
    status = "healthy" if ml_model_loaded else ("degraded" if ollama_connected else "unhealthy")
    
    return HealthResponse(
        status=status,
        ollama_connected=ollama_connected,
        database_connected=database_connected,
        ml_model_loaded=ml_model_loaded,
        ml_model_info=ml_model_info,
        validation_enabled=validation_enabled,
        validation_stats=validation_stats_data,
        cached_predictions=cached_predictions,
        model_loaded=model_loaded,
        available_models=available_models,
        drug_count=drug_count
    )


@app.post(
    "/predict",
    response_model=DrugPredictionResponse,
    tags=["Prediction"]
)
async def predict_interaction(request: DrugPredictionRequest):
    """
    Predict drug-drug interaction using hybrid ML + LLM approach with validation
    
    This endpoint:
    1. Checks cache for consistent predictions
    2. Uses trained ML model for reliable interaction prediction
    3. Validates against TWOSIDES ground truth
    4. Calibrates confidence based on validation
    5. Uses LLM (via Ollama) to generate human-readable explanations
    
    **Note**: Works with just ML model if Ollama is not available.
    """
    start_time = time.time()
    is_cached = False
    
    # Step 0: Check cache for consistent predictions
    if request.use_cache and prediction_cache:
        cached = prediction_cache.get(request.drug1, request.drug2)
        if cached:
            logger.info(f"Cache hit for {request.drug1} + {request.drug2}")
            cached_prediction = cached["prediction"]
            is_cached = True
            
            # Return cached response with updated processing time
            processing_time = (time.time() - start_time) * 1000
            return DrugPredictionResponse(
                drug1=request.drug1,
                drug2=request.drug2,
                prediction=DrugPrediction(**cached_prediction["prediction"]),
                ml_prediction=MLPrediction(**cached_prediction["ml_prediction"]) if cached_prediction.get("ml_prediction") else None,
                validation=ValidationResult(**cached_prediction["validation"]) if cached_prediction.get("validation") else None,
                prediction_source=cached_prediction.get("prediction_source", "cached"),
                is_cached=True,
                twosides_context=None,  # Don't cache context
                llm_model=cached_prediction.get("llm_model"),
                processing_time_ms=round(processing_time, 2)
            )
    
    # Get TWOSIDES context if requested and available
    twosides_context = None
    twosides_context_dict = None
    
    if request.include_context and data_service and data_service.is_connected:
        twosides_context = await data_service.get_interaction_context(
            request.drug1, 
            request.drug2
        )
        if twosides_context:
            twosides_context_dict = {
                "known_interaction": twosides_context.known_interaction,
                "side_effects": twosides_context.side_effects,
                "interaction_count": twosides_context.interaction_count
            }
    
    # Step 1: Get ML prediction (fast, reliable)
    ml_prediction_result = None
    ml_prediction_schema = None
    
    if ml_model and ml_model.is_loaded:
        ml_prediction_result = ml_model.predict(request.drug1, request.drug2)
        ml_prediction_schema = MLPrediction(
            has_interaction=ml_prediction_result["has_interaction"],
            severity=ml_prediction_result["severity"],
            confidence=ml_prediction_result["confidence"],
            model_version=ml_prediction_result.get("model_version"),
            probabilities=ml_prediction_result.get("probabilities")
        )
    
    # Step 2: Validate against ground truth
    validation_result = None
    validation_schema = None
    
    if validation_service and ml_prediction_result:
        validation_result = validation_service.validate_prediction(
            request.drug1,
            request.drug2,
            ml_prediction_result["has_interaction"],
            ml_prediction_result["severity"]
        )
        
        # Calibrate confidence
        calibrated_confidence = validation_service.calibrate_confidence(
            ml_prediction_result["confidence"],
            validation_result
        )
        
        validation_schema = ValidationResult(
            is_validated=validation_result["is_validated"],
            is_correct=validation_result.get("is_correct"),
            ground_truth_severity=validation_result.get("ground_truth", {}).get("severity") if validation_result.get("ground_truth") else None,
            calibrated_confidence=calibrated_confidence,
            message=validation_result["message"]
        )
        
        # Update ML prediction with calibrated confidence
        if ml_prediction_schema:
            ml_prediction_schema.confidence = calibrated_confidence
    
    # Step 3: Get LLM explanation (if available)
    llm_prediction = None
    prediction_source = "ml"
    
    if llm_service:
        try:
            # Include ML prediction and validation in context for better LLM explanation
            enhanced_context = twosides_context_dict.copy() if twosides_context_dict else {}
            if ml_prediction_result:
                enhanced_context["ml_prediction"] = {
                    "has_interaction": ml_prediction_result["has_interaction"],
                    "severity": ml_prediction_result["severity"],
                    "confidence": ml_prediction_result["confidence"]
                }
            if validation_result and validation_result["is_validated"]:
                enhanced_context["validated"] = validation_result["is_correct"]
            
            llm_prediction = await llm_service.predict_interaction(
                request.drug1,
                request.drug2,
                enhanced_context if enhanced_context else None
            )
            prediction_source = "hybrid" if ml_prediction_result else "llm"
        except Exception as e:
            logger.warning(f"LLM prediction failed: {e}")
    
    # Build final prediction
    if ml_prediction_result and llm_prediction:
        # Hybrid: Use ML for has_interaction/severity, LLM for explanation
        final_prediction = llm_prediction
        final_prediction.has_interaction = ml_prediction_result["has_interaction"]
        final_prediction.severity = ml_prediction_result["severity"]
        # Use calibrated confidence if available
        if validation_schema:
            final_prediction.confidence = validation_schema.calibrated_confidence
        else:
            final_prediction.confidence = ml_prediction_result["confidence"]
    elif ml_prediction_result:
        # ML only: Create basic prediction without detailed explanation
        confidence = validation_schema.calibrated_confidence if validation_schema else ml_prediction_result["confidence"]
        
        validation_note = ""
        if validation_schema and validation_schema.is_validated:
            if validation_schema.is_correct:
                validation_note = " (Validated against database ✓)"
            else:
                validation_note = " (Conflicts with database record)"
        
        final_prediction = DrugPrediction(
            has_interaction=ml_prediction_result["has_interaction"],
            severity=ml_prediction_result["severity"],
            confidence=confidence,
            explanation=f"ML model predicts {'an interaction' if ml_prediction_result['has_interaction'] else 'no significant interaction'} between {request.drug1} and {request.drug2} with {ml_prediction_result['severity']} severity.{validation_note}",
            mechanism="Consult a healthcare professional for detailed mechanism information.",
            recommendations=[
                "Consult your healthcare provider before combining medications",
                "Review drug package inserts for detailed interaction information",
                "Report any unusual symptoms to your doctor"
            ],
            reasoning=f"Prediction based on ML model trained on drug interaction data. Confidence: {confidence:.1%}"
        )
    elif llm_prediction:
        # LLM only
        final_prediction = llm_prediction
    else:
        # Neither available
        raise HTTPException(
            status_code=503,
            detail="Neither ML model nor LLM service available. Train the model with train_model.py or ensure Ollama is running."
        )
    
    processing_time = (time.time() - start_time) * 1000
    
    # Build response
    response = DrugPredictionResponse(
        drug1=request.drug1,
        drug2=request.drug2,
        prediction=final_prediction,
        ml_prediction=ml_prediction_schema,
        validation=validation_schema,
        prediction_source=prediction_source,
        is_cached=is_cached,
        twosides_context=twosides_context,
        llm_model=llm_service.model if llm_service else None,
        processing_time_ms=round(processing_time, 2)
    )
    
    # Cache the prediction for consistency
    if request.use_cache and prediction_cache and not is_cached:
        cache_data = {
            "prediction": final_prediction.model_dump(),
            "ml_prediction": ml_prediction_schema.model_dump() if ml_prediction_schema else None,
            "validation": validation_schema.model_dump() if validation_schema else None,
            "prediction_source": prediction_source,
            "llm_model": llm_service.model if llm_service else None
        }
        prediction_cache.set(request.drug1, request.drug2, cache_data)
    
    return response


@app.get(
    "/predict/ml",
    response_model=MLPrediction,
    tags=["Prediction"]
)
async def predict_ml_only(
    drug1: str = Query(..., min_length=1, description="First drug name"),
    drug2: str = Query(..., min_length=1, description="Second drug name")
):
    """
    Get pure ML model prediction (without LLM explanation)
    
    Fast endpoint for quick predictions without waiting for LLM.
    """
    if not ml_model or not ml_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run train_model.py to train the model."
        )
    
    result = ml_model.predict(drug1, drug2)
    
    # Validate and calibrate
    if validation_service:
        validation_result = validation_service.validate_prediction(
            drug1, drug2, result["has_interaction"], result["severity"]
        )
        calibrated_confidence = validation_service.calibrate_confidence(
            result["confidence"], validation_result
        )
        result["confidence"] = calibrated_confidence
    
    return MLPrediction(
        has_interaction=result["has_interaction"],
        severity=result["severity"],
        confidence=result["confidence"],
        model_version=result.get("model_version"),
        probabilities=result.get("probabilities")
    )


@app.get(
    "/drugs/search",
    response_model=DrugSearchResponse,
    tags=["Drugs"]
)
async def search_drugs(
    query: str = Query(..., min_length=1, description="Drug name to search"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results")
):
    """
    Search for drugs in the database
    """
    if not data_service or not data_service.is_connected:
        return DrugSearchResponse(
            query=query,
            results=[],
            total_count=0
        )
    
    results = await data_service.search_drugs(query, limit)
    
    return DrugSearchResponse(
        query=query,
        results=results,
        total_count=len(results)
    )


@app.get("/models", tags=["Models"])
async def list_models():
    """List available models (ML and LLM)"""
    response = {
        "ml_model": None,
        "llm_models": [],
        "current_llm": None
    }
    
    if ml_model and ml_model.is_loaded:
        response["ml_model"] = ml_model.get_model_info()
    
    if llm_service:
        response["llm_models"] = llm_service.available_models
        response["current_llm"] = llm_service.model
    
    return response


@app.get("/model/info", response_model=ModelInfo, tags=["Models"])
async def get_model_info():
    """Get detailed ML model information"""
    if not ml_model:
        return ModelInfo(is_loaded=False)
    
    info = ml_model.get_model_info()
    return ModelInfo(
        is_loaded=info.get("is_loaded", False),
        version=info.get("version"),
        accuracy=info.get("accuracy"),
        training_date=info.get("training_date"),
        num_training_samples=info.get("num_training_samples"),
        model_type=info.get("model_type")
    )


@app.get("/validation/stats", response_model=ValidationStats, tags=["Validation"])
async def get_validation_stats():
    """Get validation service statistics"""
    if not validation_service:
        return ValidationStats()
    
    stats = validation_service.get_validation_stats()
    return ValidationStats(
        total_validated=stats["total_validated"],
        correct_predictions=stats["correct_predictions"],
        false_positives=stats["false_positives"],
        false_negatives=stats["false_negatives"],
        accuracy=stats["accuracy"],
        ground_truth_count=stats["ground_truth_count"]
    )


@app.delete("/cache", tags=["Cache"])
async def clear_cache():
    """Clear prediction cache"""
    if prediction_cache:
        prediction_cache.clear()
        return {"message": "Cache cleared", "cached_predictions": 0}
    return {"message": "No cache available"}


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if logger.level == logging.DEBUG else None,
            error_code="INTERNAL_ERROR"
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True
    )
