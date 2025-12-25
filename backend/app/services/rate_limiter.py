"""
Simple rate limiting using Redis (fixed window).
"""
import logging
from typing import Optional

from fastapi import HTTPException, Request, status

from app.services.cache import get_redis_client

logger = logging.getLogger(__name__)


async def rate_limit(request: Request, limit: int = 60, window_seconds: int = 60, key_prefix: str = "rl") -> None:
    """
    Apply a fixed-window rate limit based on client IP.

    Args:
        request: FastAPI request
        limit: allowed requests per window
        window_seconds: window size in seconds
        key_prefix: redis key prefix
    """
    client_ip = request.client.host if request.client else "unknown"
    key = f"{key_prefix}:{client_ip}"

    client = await get_redis_client()
    if not client:
        return  # fail-open if no redis

    try:
        current = await client.incr(key)
        if current == 1:
            await client.expire(key, window_seconds)
        if current > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
            )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network dependent
        logger.debug(f"Rate limit check failed: {exc}")
        return

