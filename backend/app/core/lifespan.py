"""Backward-compatible lifespan exports from services layer."""
from app.services.app_lifespan import lifespan, seed_initial_data, validate_startup_configuration

__all__ = ["lifespan", "seed_initial_data", "validate_startup_configuration"]
