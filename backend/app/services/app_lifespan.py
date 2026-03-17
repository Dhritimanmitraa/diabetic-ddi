"""FastAPI lifespan and startup validation helpers."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.database import init_db

logger = logging.getLogger(__name__)


def validate_startup_configuration() -> None:
    """Optionally fail fast in production when required secrets are missing."""
    settings = get_settings()
    is_production = settings.APP_ENV.lower() in {"prod", "production"}
    if not is_production:
        return

    if not settings.STRICT_STARTUP_VALIDATION:
        logger.warning("Production strict startup validation is disabled")
        return

    missing: list[str] = []
    insecure: list[str] = []

    if settings.REQUIRE_API_KEY_FOR_ADMIN and not settings.API_KEY:
        missing.append("API_KEY")

    if settings.REQUIRE_GEMINI_KEY and not (
        settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
    ):
        missing.append("GEMINI_API_KEY/GOOGLE_API_KEY")

    if settings.ENABLE_CLOUD_SPEECH and not settings.CLOUD_SPEECH_API_KEY:
        missing.append("CLOUD_SPEECH_API_KEY")

    if not settings.JWT_SECRET or settings.JWT_SECRET == "change-me-in-production":
        insecure.append("JWT_SECRET")

    if settings.DB_AUTO_CREATE:
        logger.warning(
            "DB_AUTO_CREATE is enabled in production. Prefer Alembic migrations during deployment."
        )

    if missing:
        logger.error(
            "Startup validation failed",
            extra={"missing_settings": missing, "app_env": settings.APP_ENV},
        )
        raise RuntimeError(f"Missing required production settings: {', '.join(missing)}")

    if insecure:
        logger.error(
            "Startup validation failed due to insecure settings",
            extra={"insecure_settings": insecure, "app_env": settings.APP_ENV},
        )
        raise RuntimeError(f"Insecure production settings: {', '.join(insecure)}")


async def seed_initial_data() -> None:
    """Placeholder for seeding data if needed."""


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Application lifespan handler."""
    validate_startup_configuration()
    logger.info("Initializing database metadata")
    await init_db()
    logger.info("Database initialization complete")

    await seed_initial_data()

    from app.ml.scheduler import start_scheduler, stop_scheduler

    start_scheduler()
    try:
        yield
    finally:
        stop_scheduler()
        logger.info("Shutting down")
        from app.services.api_client import close_api_client

        await close_api_client()
