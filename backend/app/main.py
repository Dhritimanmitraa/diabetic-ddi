"""FastAPI application entry point."""
from __future__ import annotations

import logging

from fastapi import APIRouter, FastAPI

from app.api.health import router as health_router
from app.api.router import include_api_routers
from app.config import get_settings
from app.routers.auth import router as auth_router
from app.routers.metrics_router import router as metrics_router
from app.services.app_lifespan import lifespan
from app.services.http_middleware import setup_http_middleware
from app.services.logging_setup import configure_logging

configure_logging()

logger = logging.getLogger(__name__)
settings = get_settings()

APP_DESCRIPTION = """
## Drug-Drug Interaction Prediction System

A machine learning-powered system to check drug interactions and find safe alternatives.

### Features:
- **Check Interactions**: Verify if two drugs are safe to use together
- **Image Recognition**: Extract drug names from photos using OCR
- **Alternative Suggestions**: Get safe alternative medications
- **Comprehensive Database**: 100,000+ drug interactions

### Severity Levels:
- Minor: Generally safe, minimal effects
- Moderate: Use caution, monitor for effects
- Major: Significant interaction, consult healthcare provider
- Contraindicated: Do NOT use together
"""


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=APP_DESCRIPTION,
        lifespan=lifespan,
    )

    setup_http_middleware(app)
    include_api_routers(app)
    v1_auth_router = APIRouter(prefix="/v1")
    v1_auth_router.include_router(auth_router)
    v1_metrics_router = APIRouter(prefix="/v1")
    v1_metrics_router.include_router(metrics_router)
    app.include_router(v1_auth_router)
    app.include_router(v1_metrics_router)
    app.include_router(auth_router)
    app.include_router(metrics_router)
    app.include_router(health_router)

    return app


app = create_app()
