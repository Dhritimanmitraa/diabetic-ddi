"""Shared authentication dependencies."""
from fastapi import Header, HTTPException, status


def require_api_key(x_api_key: str | None = Header(None)) -> None:
    from app.config import get_settings
    settings = get_settings()
    if not settings.REQUIRE_API_KEY_FOR_ADMIN:
        return

    expected = settings.API_KEY
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API key protection is enabled but API_KEY is not configured",
        )
    if x_api_key == expected:
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )
