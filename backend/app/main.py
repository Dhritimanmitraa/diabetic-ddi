"""
Drug-Drug Interaction Prediction API

Main FastAPI application entry point.
"""
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from contextlib import asynccontextmanager
from typing import List, Optional
import logging
from datetime import datetime

from app.config import get_settings
from app.services.rate_limiter import rate_limit
from app.services.auth import require_api_key
from app.services.tasks import enqueue_training, enqueue_data_refresh, get_job
from app.database import init_db, get_db, engine, Base
from app.models import Drug, DrugInteraction, Category, ComparisonLog
from app.schemas import (
    DrugSearch, DrugResponse, DrugCreate,
    InteractionCheckRequest, InteractionCheckResponse,
    AlternativeSuggestionResponse,
    OCRRequest, OCRResponse,
    DatabaseStats,
    SeverityLevel,
)
from app.services import (
    create_interaction_service,
    create_ocr_service,
    create_comparison_logger
)
from app.diabetic.router import router as diabetic_router
from app.exceptions import ValidationError as ValidationException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized successfully!")
    
    # Seed initial data if empty
    await seed_initial_data()
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## Drug-Drug Interaction Prediction System
    
    A machine learning-powered system to check drug interactions and find safe alternatives.
    
    ### Features:
    - **Check Interactions**: Verify if two drugs are safe to use together
    - **Image Recognition**: Extract drug names from photos using OCR
    - **Alternative Suggestions**: Get safe alternative medications
    - **Comprehensive Database**: 100,000+ drug interactions
    
    ### Severity Levels:
    - 🟢 **Minor**: Generally safe, minimal effects
    - 🟡 **Moderate**: Use caution, monitor for effects
    - 🟠 **Major**: Significant interaction, consult healthcare provider
    - 🔴 **Contraindicated**: Do NOT use together
    """,
    lifespan=lifespan
)

# Log drug search/list requests visibly in terminal
@app.middleware("http")
async def log_drug_requests(request: Request, call_next):
    """
    Log all /drugs and /drugs/search calls to console for visibility.
    """
    path = request.url.path
    if path.startswith("/drugs"):
        client_ip = request.client.host if request.client else "unknown"
        info = {
            "event": "http_request",
            "path": path,
            "method": request.method,
            "query": dict(request.query_params),
            "client_ip": client_ip,
        }
        logger.info(info)
        # Explicit print to ensure it shows in terminal output
        print(f"[search-log] {request.method} {path} query={info['query']} ip={client_ip}")
    response = await call_next(request)
    return response

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include diabetic patient module router
app.include_router(diabetic_router)


# ============== Health & Stats Endpoints ==============

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check."""
    return {
        "message": "Drug-Drug Interaction Prediction API",
        "version": settings.APP_VERSION,
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION
    }


@app.get("/health/live", tags=["Health"])
async def liveness():
    """Simple liveness probe."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def readiness(db: AsyncSession = Depends(get_db)):
    """
    Readiness probe that checks database connectivity and model availability.
    """
    # DB check
    await db.execute(select(func.count(Drug.id)))

    # Model availability check (lightweight)
    import os
    model_dir = "./models"
    has_models = all(
        os.path.exists(os.path.join(model_dir, fname))
        for fname in [
            "feature_extractor.pkl",
            "random_forest_model.pkl",
            "xgboost_model.pkl",
            "lightgbm_model.pkl",
        ]
    )

    return {
        "status": "ready" if has_models else "degraded",
        "models_loaded": has_models,
    }


@app.get("/stats", response_model=DatabaseStats, tags=["Statistics"])
async def get_statistics(db: AsyncSession = Depends(get_db)):
    """Get database statistics."""
    # Count drugs
    drug_count = await db.execute(select(func.count(Drug.id)))
    total_drugs: int = drug_count.scalar() or 0
    
    # Count interactions
    interaction_count = await db.execute(select(func.count(DrugInteraction.id)))
    total_interactions: int = interaction_count.scalar() or 0
    
    # Count by severity
    severity_counts = {}
    for severity in ["minor", "moderate", "major", "contraindicated"]:
        count = await db.execute(
            select(func.count(DrugInteraction.id)).where(
                DrugInteraction.severity == severity
            )
        )
        severity_counts[severity] = count.scalar()
    
    return DatabaseStats(
        total_drugs=total_drugs,
        total_interactions=total_interactions,
        interactions_by_severity=severity_counts,
        last_updated=datetime.utcnow()
    )


# ============== Drug Search Endpoints ==============

@app.get("/drugs", response_model=List[DrugResponse], tags=["Drugs"])
async def list_drugs(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """
    List all drugs with pagination.
    
    Returns drugs ordered by name for browsing.
    """
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        {
            "event": "drug_browse",
            "path": "/drugs",
            "limit": limit,
            "offset": offset,
            "client_ip": client_ip,
        }
    )
    result = await db.execute(
        select(Drug)
        .order_by(Drug.name)
        .offset(offset)
        .limit(min(limit, 100))  # Max 100 per request
    )
    drugs = result.scalars().all()
    return [DrugResponse.model_validate(d) for d in drugs]


@app.get("/drugs/search", response_model=List[DrugResponse], tags=["Drugs"])
async def search_drugs(
    request: Request,
    query: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Search for drugs by name.
    
    Supports partial matching on drug name, generic name, and brand names.
    """
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        {
            "event": "drug_search",
            "path": "/drugs/search",
            "query": query,
            "limit": limit,
            "client_ip": client_ip,
        }
    )
    service = create_interaction_service(db)
    drugs = await service.search_drugs(query, limit)
    return [DrugResponse.model_validate(d) for d in drugs]


@app.get("/drugs/{drug_id}", response_model=DrugResponse, tags=["Drugs"])
async def get_drug(drug_id: int, db: AsyncSession = Depends(get_db)):
    """Get drug details by ID."""
    result = await db.execute(select(Drug).where(Drug.id == drug_id))
    drug = result.scalar_one_or_none()
    
    if not drug:
        raise HTTPException(status_code=404, detail="Drug not found")
    
    return DrugResponse.model_validate(drug)


@app.get("/drugs/name/{drug_name}", response_model=DrugResponse, tags=["Drugs"])
async def get_drug_by_name(drug_name: str, db: AsyncSession = Depends(get_db)):
    """Get drug details by name."""
    service = create_interaction_service(db)
    drug = await service.get_drug_by_name(drug_name)
    
    if not drug:
        raise HTTPException(status_code=404, detail="Drug not found")
    
    return DrugResponse.model_validate(drug)


# ============== Interaction Check Endpoints ==============

@app.post("/interactions/check", response_model=InteractionCheckResponse, tags=["Interactions"])
async def check_interaction(
    request: InteractionCheckRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Check if two drugs have a known interaction.
    
    Returns interaction details, safety status, and recommendations.
    All comparisons are logged for tracking.
    """
    service = create_interaction_service(db)

    # Rules-based result (for backstop and details)
    result = await service.check_interaction(request.drug1_name, request.drug2_name)

    # ML primary inference
    ml_probability = None
    ml_severity = None
    ml_predicted = None
    decision_source = "rules_only"
    try:
        from app.ml.predictor import get_predictor
        drug1 = await service.get_drug_by_name(request.drug1_name)
        drug2 = await service.get_drug_by_name(request.drug2_name)
        if drug1 and drug2:
            drug1_dict = {
                'name': drug1.name,
                'generic_name': drug1.generic_name,
                'drug_class': drug1.drug_class,
                'description': drug1.description,
                'mechanism': drug1.mechanism,
                'indication': drug1.indication,
                'molecular_weight': drug1.molecular_weight,
                'is_approved': drug1.is_approved,
            }
            drug2_dict = {
                'name': drug2.name,
                'generic_name': drug2.generic_name,
                'drug_class': drug2.drug_class,
                'description': drug2.description,
                'mechanism': drug2.mechanism,
                'indication': drug2.indication,
                'molecular_weight': drug2.molecular_weight,
                'is_approved': drug2.is_approved,
            }
            predictor = get_predictor("./models")
            if predictor.is_loaded:
                ml_res = predictor.predict(drug1_dict, drug2_dict)
                ml_probability = ml_res.interaction_probability
                ml_severity = ml_res.severity_prediction
                ml_predicted = ml_res.predicted_interaction
                decision_source = "ml_primary"
    except Exception as e:
        logger.error(f"ML prediction failed in /interactions/check: {e}")

    # Hybrid gate: rules override for high-risk constraints
    rule_override_reason = None
    if result.interaction and result.interaction.severity in [SeverityLevel.CONTRAINDICATED, SeverityLevel.MAJOR]:
        final_has = True
        final_safe = False
        decision_source = "rule_override"
        rule_override_reason = f"Rule flagged {result.interaction.severity} severity interaction"
    elif ml_predicted is not None:
        final_has = bool(ml_predicted)
        final_safe = not final_has
    else:
        final_has = result.has_interaction
        final_safe = result.is_safe

    # Attach ML fields
    result.has_interaction = final_has
    result.is_safe = final_safe
    result.ml_probability = ml_probability
    result.ml_severity = ml_severity
    result.ml_decision_source = decision_source
    
    # Log the comparison with ML audit fields
    comparison_logger = create_comparison_logger(db)
    await comparison_logger.log_comparison(
        drug1_name=request.drug1_name,
        drug2_name=request.drug2_name,
        drug1_id=result.drug1.id if result.drug1.id != 0 else None,
        drug2_id=result.drug2.id if result.drug2.id != 0 else None,
        has_interaction=result.has_interaction,
        is_safe=result.is_safe,
        severity=result.interaction.severity if result.interaction else None,
        effect=result.interaction.effect if result.interaction else None,
        safety_message=result.safety_message,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("user-agent"),
        # ML audit fields
        ml_probability=ml_probability,
        ml_severity=ml_severity,
        ml_decision_source=decision_source,
        rule_override_reason=rule_override_reason
    )
    
    return result


@app.get("/interactions/check/{drug1}/{drug2}", response_model=InteractionCheckResponse, tags=["Interactions"])
async def check_interaction_get(
    drug1: str,
    drug2: str,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Check interaction between two drugs (GET endpoint).
    
    URL-encoded drug names. All comparisons are logged.
    """
    service = create_interaction_service(db)
    result = await service.check_interaction(drug1, drug2)
    
    # Log the comparison
    comparison_logger = create_comparison_logger(db)
    await comparison_logger.log_comparison(
        drug1_name=drug1,
        drug2_name=drug2,
        drug1_id=result.drug1.id if result.drug1.id != 0 else None,
        drug2_id=result.drug2.id if result.drug2.id != 0 else None,
        has_interaction=result.has_interaction,
        is_safe=result.is_safe,
        severity=result.interaction.severity if result.interaction else None,
        effect=result.interaction.effect if result.interaction else None,
        safety_message=result.safety_message,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("user-agent")
    )
    
    return result


@app.get("/interactions/drug/{drug_name}", tags=["Interactions"])
async def get_drug_interactions(
    drug_name: str,
    severity: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get all known interactions for a specific drug.
    
    Optionally filter by severity level.
    """
    service = create_interaction_service(db)
    interactions = await service.get_all_interactions_for_drug(drug_name, severity)
    
    return {
        "drug": drug_name,
        "total_interactions": len(interactions),
        "interactions": [
            {
                "id": i.id,
                "other_drug": i.drug2.name if i.drug1.name.upper() == drug_name.upper() else i.drug1.name,
                "severity": i.severity,
                "effect": i.effect,
                "management": i.management
            }
            for i in interactions
        ]
    }


# ============== Alternative Suggestions Endpoints ==============

@app.post("/alternatives", response_model=AlternativeSuggestionResponse, tags=["Alternatives"])
async def get_alternatives(
    request: InteractionCheckRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get safe alternative medications when an interaction is detected.
    
    Finds similar drugs that don't interact with the other medication.
    """
    service = create_interaction_service(db)
    
    try:
        return await service.find_alternatives(
            request.drug1_name,
            request.drug2_name
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/alternatives/{drug1}/{drug2}", response_model=AlternativeSuggestionResponse, tags=["Alternatives"])
async def get_alternatives_get(
    drug1: str,
    drug2: str,
    db: AsyncSession = Depends(get_db)
):
    """Get safe alternatives (GET endpoint)."""
    service = create_interaction_service(db)
    
    try:
        return await service.find_alternatives(drug1, drug2)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============== OCR Endpoints ==============

@app.post("/ocr/extract", response_model=OCRResponse, tags=["OCR"])
async def extract_from_image(
    request: OCRRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Extract drug names from an image using OCR.
    
    Accepts base64 encoded image of medication labels, bottles, or prescriptions.
    """
    ocr_service = create_ocr_service(settings.TESSERACT_CMD)
    
    try:
        raw_text, detected_drugs, confidence = ocr_service.extract_from_base64(
            request.image_base64
        )
        
        # Try to match detected drugs with database with fuzzy matching
        service = create_interaction_service(db)
        matched_drugs = []
        
        logger.info(f"Attempting to match {len(detected_drugs)} detected drug names with database")
        
        for drug_name in detected_drugs:
            # First try exact/partial match
            drugs = await service.search_drugs(drug_name, limit=5)
            
            if drugs:
                # Use the first match (most relevant)
                matched_drugs.append(drugs[0].name)
                logger.info(f"Matched '{drug_name}' -> '{drugs[0].name}'")
            else:
                # Try fuzzy matching using OCR service
                # Get all drugs from database for fuzzy matching (limit to avoid memory issues)
                all_drugs_stmt = select(Drug.name).limit(10000)  # Limit for performance
                all_drugs_result = await db.execute(all_drugs_stmt)
                all_drug_names = [d[0] for d in all_drugs_result.fetchall()]
                
                if all_drug_names:
                    fuzzy_matches = ocr_service.find_similar_drug_names(
                        drug_name, all_drug_names, threshold=0.5  # Lower threshold for OCR errors
                    )
                    if fuzzy_matches:
                        matched_drugs.append(fuzzy_matches[0][0])
                        logger.info(f"Fuzzy matched '{drug_name}' -> '{fuzzy_matches[0][0]}' (score: {fuzzy_matches[0][1]:.2f})")
                    else:
                        # Keep original if no match found
                        matched_drugs.append(drug_name)
                        logger.warning(f"No match found for '{drug_name}'")
                else:
                    matched_drugs.append(drug_name)
        
        return OCRResponse(
            extracted_text=raw_text[:1000],  # Limit text length
            detected_drugs=matched_drugs,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"OCR error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/ocr/upload", response_model=OCRResponse, tags=["OCR"])
async def extract_from_upload(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Extract drug names from uploaded image file.
    
    Accepts image files (JPEG, PNG).
    """
    import base64
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise ValidationException("File must be an image (JPEG, PNG, etc.)")
    
    # Read and encode image
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
    
    # Process using the base64 endpoint
    request = OCRRequest(image_base64=base64_image)
    return await extract_from_image(request, db)


# ============== Comparison History Endpoints ==============

@app.get("/history", tags=["History"])
async def get_comparison_history(
    limit: int = 50,
    offset: int = 0,
    severity: Optional[str] = None,
    search: Optional[str] = None,
    is_safe: Optional[bool] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get drug comparison history with pagination and filtering.
    """
    limit = min(max(limit, 1), 100)
    offset = max(offset, 0)

    comparison_logger = create_comparison_logger(db)
    comparisons, total = await comparison_logger.get_comparisons(
        limit=limit,
        offset=offset,
        severity=severity,
        search=search,
        is_safe=is_safe,
    )
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "comparisons": [
            {
                "id": c.id,
                "timestamp": c.timestamp.isoformat(),
                "drug1": c.drug1_name,
                "drug2": c.drug2_name,
                "has_interaction": c.has_interaction,
                "is_safe": c.is_safe,
                "severity": c.severity,
                "effect": c.effect,
                "safety_message": c.safety_message
            }
            for c in comparisons
        ]
    }


@app.get("/history/stats", tags=["History"])
async def get_comparison_stats(db: AsyncSession = Depends(get_db)):
    """
    Get statistics about all comparisons made.
    
    Returns total counts, safe/unsafe breakdown, and severity distribution.
    """
    comparison_logger = create_comparison_logger(db)
    stats = await comparison_logger.get_comparison_stats()
    return stats


@app.get("/history/dangerous", tags=["History"])
async def get_dangerous_combinations(db: AsyncSession = Depends(get_db)):
    """
    Get all dangerous (major/contraindicated) combinations that users have checked.
    
    Useful for identifying common dangerous drug pairs being queried.
    """
    comparison_logger = create_comparison_logger(db)
    dangerous = await comparison_logger.get_dangerous_combinations_found()
    
    return {
        "total_dangerous_checks": len(dangerous),
        "combinations": dangerous
    }


@app.get("/history/popular-drugs", tags=["History"])
async def get_most_checked_drugs(
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the most frequently checked drugs.
    
    Shows which drugs users are most concerned about.
    """
    comparison_logger = create_comparison_logger(db)
    popular = await comparison_logger.get_most_checked_drugs(limit)
    
    return {
        "most_checked_drugs": popular
    }


# ============== ML Prediction Endpoints ==============

@app.post("/ml/predict", tags=["Machine Learning"])
async def ml_predict_interaction(
    request: InteractionCheckRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Predict drug interaction using ML models.
    
    Uses trained Random Forest, XGBoost, and LightGBM models
    with Bayesian-optimized hyperparameters.
    
    Returns probability of interaction and severity prediction.
    """
    from app.ml.predictor import get_predictor
    
    # Get drug information from database
    service = create_interaction_service(db)
    drug1 = await service.get_drug_by_name(request.drug1_name)
    drug2 = await service.get_drug_by_name(request.drug2_name)
    
    if not drug1 or not drug2:
        return {
            "error": "One or both drugs not found in database",
            "drug1_found": drug1 is not None,
            "drug2_found": drug2 is not None
        }
    
    # Convert to dict for predictor
    drug1_dict = {
        'name': drug1.name,
        'generic_name': drug1.generic_name,
        'drug_class': drug1.drug_class,
        'description': drug1.description,
        'mechanism': drug1.mechanism,
        'indication': drug1.indication,
        'molecular_weight': drug1.molecular_weight,
        'is_approved': drug1.is_approved,
    }
    
    drug2_dict = {
        'name': drug2.name,
        'generic_name': drug2.generic_name,
        'drug_class': drug2.drug_class,
        'description': drug2.description,
        'mechanism': drug2.mechanism,
        'indication': drug2.indication,
        'molecular_weight': drug2.molecular_weight,
        'is_approved': drug2.is_approved,
    }
    
    try:
        predictor = get_predictor("./models")
        if not predictor.is_loaded:
            return {
                "error": "ML models not loaded. Please train models first.",
                "hint": "Run: python -m scripts.train_models"
            }
        
        result = predictor.predict(drug1_dict, drug2_dict)
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"ML prediction error: {e}")
        return {
            "error": f"Prediction failed: {str(e)}",
            "hint": "Ensure models are trained and loaded"
        }


@app.get("/ml/model-info", tags=["Machine Learning"])
async def get_ml_model_info():
    """
    Get information about trained ML models.
    
    Returns model metrics, training parameters, and feature importance.
    """
    from app.ml.predictor import get_predictor
    
    try:
        predictor = get_predictor("./models")
        
        if not predictor.is_loaded:
            return {
                "status": "not_loaded",
                "message": "ML models not yet trained",
                "hint": "Run: python -m scripts.train_models"
            }
        
        info = predictor.get_model_info()
        feature_importance = predictor.get_feature_importance()
        
        return {
            "status": "loaded",
            "models": info,
            "feature_importance": feature_importance
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/ml/comparison", tags=["Machine Learning"])
async def get_optimization_comparison():
    """
    Get comparison results of optimization methods.
    
    Shows performance comparison between:
    - Bayesian Optimization (TPE)
    - Grid Search
    - Random Search
    """
    import os
    import json
    
    model_dir = "./models"
    comparison_files = []
    
    for model_type in ["random_forest", "xgboost", "lightgbm"]:
        filepath = os.path.join(model_dir, f"{model_type}_comparison.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                comparison_files.append({
                    "model": model_type,
                    "comparison": json.load(f)
                })
    
    if not comparison_files:
        return {
            "status": "no_comparisons",
            "message": "No optimization comparison results found",
            "hint": "Run training with comparison enabled"
        }
    
    # Calculate overall summary
    bayesian_wins = 0
    total_comparisons = 0
    avg_efficiency_gain = []
    
    for comp in comparison_files:
        summary = comp["comparison"].get("comparison_summary", {})
        if summary.get("winner") == "bayesian":
            bayesian_wins += 1
        total_comparisons += 1
        
        efficiency = summary.get("efficiency_gain", {})
        if efficiency.get("trial_reduction_percent"):
            avg_efficiency_gain.append(efficiency["trial_reduction_percent"])
    
    return {
        "status": "loaded",
        "total_models_compared": total_comparisons,
        "bayesian_wins": bayesian_wins,
        "average_trial_reduction_percent": sum(avg_efficiency_gain) / len(avg_efficiency_gain) if avg_efficiency_gain else 0,
        "detailed_comparisons": comparison_files
    }


@app.post("/ml/train", tags=["Machine Learning"])
async def trigger_model_training(
    request: Request,
    n_trials: int = 50,
    run_comparison: bool = True,
    api_key: None = Depends(require_api_key)
):
    """
    Enqueue ML model training (background job).
    Requires API key and rate limiting.
    """
    await rate_limit(request, limit=settings.RATE_LIMIT_REQUESTS_PER_MIN, key_prefix="ml-train")
    job = enqueue_training(n_trials=n_trials, run_comparison=run_comparison)
    return {"status": "queued", "job_id": job.id}


@app.post("/jobs/fetch-data", tags=["Jobs"])
async def enqueue_data_fetch(
    request: Request,
    drugs: int = 5000,
    interactions: int = 100000,
    api_key: None = Depends(require_api_key)
):
    """Enqueue real data refresh job."""
    await rate_limit(request, limit=settings.RATE_LIMIT_REQUESTS_PER_MIN, key_prefix="data-refresh")
    job = enqueue_data_refresh(drugs=drugs, interactions=interactions)
    return {"status": "queued", "job_id": job.id}


@app.get("/jobs/{job_id}", tags=["Jobs"])
async def get_job_status(job_id: str, api_key: None = Depends(require_api_key)):
    """Get status of a background job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job.get_status(),
        "enqueued_at": job.enqueued_at,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "result": job.result if job.is_finished else None,
    }


# ============== Data Loading ==============

async def seed_initial_data():
    """Check if database has data. If empty, prompt user to fetch from APIs."""
    async with engine.begin() as conn:
        # Check if we have drugs
        result = await conn.execute(select(func.count(Drug.id)))
        count: int = result.scalar() or 0
        
        if count > 0:
            logger.info(f"Database has {count} drugs loaded from real APIs.")
            return
    
    logger.warning("=" * 60)
    logger.warning("DATABASE IS EMPTY - No drug data loaded yet!")
    logger.warning("=" * 60)
    logger.warning("To fetch REAL data from FDA and NIH APIs, run:")
    logger.warning("  python -m scripts.fetch_real_data")
    logger.warning("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

