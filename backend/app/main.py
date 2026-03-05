"""
Drug-Drug Interaction Prediction API

Main FastAPI application entry point.
"""
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from contextlib import asynccontextmanager
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.database import init_db, get_db
from app.models import Drug, DrugInteraction
from app.schemas import DatabaseStats
from app.services.cache import get_cached_db_stats, cache_db_stats
import os

# Import Routers
from app.diabetic.router import router as diabetic_router
from app.prescription.router import router as prescription_router
from app.routers.drugs import router as drugs_router
from app.routers.interactions import router as interactions_router
from app.routers.ocr import router as ocr_router
from app.routers.history import router as history_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

settings = get_settings()

async def seed_initial_data():
    """Placeholder for seeding data if needed."""
    pass

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
    from app.services.api_client import close_api_client
    await close_api_client()


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
    allow_credentials=False,  # Cannot use True with wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(diabetic_router)
app.include_router(prescription_router)
app.include_router(drugs_router)
app.include_router(interactions_router)
app.include_router(ocr_router)
app.include_router(history_router)


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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    model_dir = str(Path(__file__).resolve().parent.parent / "models")
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


@app.get("/health/apis", tags=["Health"])
async def api_health_status():
    """
    Get health status of all external APIs used for drug data.
    
    Shows circuit breaker states, cache statistics, and API availability.
    """
    try:
        from app.services.api_client import get_api_client
        
        client = get_api_client()
        status = client.get_health_status()
        
        # Add LLM status
        llm_status = {
            "ollama_configured": bool(settings.OLLAMA_HOST),
            "gemini_configured": bool(settings.GOOGLE_API_KEY),
            "fallback_to_templates": settings.LLM_FALLBACK_TO_TEMPLATES,
        }
        
        # Add API key status (without exposing keys)
        api_keys = {
            "openfda_key_configured": bool(settings.OPENFDA_API_KEY),
            "umls_key_configured": bool(settings.UMLS_API_KEY),
        }
        
        return {
            "status": "healthy",
            "external_apis": status,
            "llm_services": llm_status,
            "api_keys": api_keys,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except Exception as e:
        logger.error(f"API health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

@app.get("/stats", response_model=DatabaseStats, tags=["Statistics"])
async def get_statistics(db: AsyncSession = Depends(get_db)):
    """
    Get database statistics.
    
    Results are cached for 5 minutes to reduce database load.
    """
    # Try cache first
    cached_stats = await get_cached_db_stats()
    if cached_stats is not None:
        logger.debug("Cache HIT for database stats")
        return DatabaseStats(**cached_stats)
    
    logger.debug("Cache MISS for database stats")
    
    # Count drugs
    drug_count = await db.execute(select(func.count(Drug.id)))
    total_drugs = drug_count.scalar()
    
    # Count interactions
    interaction_count = await db.execute(select(func.count(DrugInteraction.id)))
    total_interactions = interaction_count.scalar()
    
    # Count by severity using single GROUP BY query
    severity_query = await db.execute(
        select(DrugInteraction.severity, func.count(DrugInteraction.id))
        .group_by(DrugInteraction.severity)
    )
    severity_results = severity_query.all()
    severity_counts = {row[0]: row[1] for row in severity_results if row[0]}
    
    stats_data = {
        "total_drugs": total_drugs,
        "total_interactions": total_interactions,
        "interactions_by_severity": severity_counts,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    
    # Cache the results
    await cache_db_stats(stats_data)
    
    return DatabaseStats(
        total_drugs=total_drugs,
        total_interactions=total_interactions,
        interactions_by_severity=severity_counts,
        last_updated=datetime.now(timezone.utc)
    )
