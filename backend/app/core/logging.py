"""Backward-compatible logging export from services layer."""
from app.services.logging_setup import configure_logging

__all__ = ["configure_logging"]
