"""
Redis cache utilities.
Uses async redis client when REDIS_URL is configured. Falls back gracefully if
Redis is unavailable.
"""

import json
import logging
import os
from typing import Any, Optional

import redis.asyncio as redis  # type: ignore

logger = logging.getLogger(__name__)

_redis_client: Optional[redis.Redis] = None


async def get_redis_client() -> Optional[redis.Redis]:
    """Return a shared async Redis client if REDIS_URL is set and reachable."""
    global _redis_client
    if _redis_client:
        return _redis_client

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        _redis_client = redis.from_url(
            redis_url, encoding="utf-8", decode_responses=True
        )
        # quick ping to validate connection
        await _redis_client.ping()
        logger.info("Connected to Redis cache")
        return _redis_client
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning(f"Redis not available ({exc}); caching disabled")
        _redis_client = None
        return None


async def cache_get_json(key: str) -> Optional[Any]:
    client = await get_redis_client()
    if not client:
        return None
    try:
        data = await client.get(key)
        return json.loads(data) if data else None
    except Exception as exc:  # pragma: no cover - network dependent
        logger.debug(f"Cache get failed for {key}: {exc}")
        return None


async def cache_set_json(key: str, value: Any, ttl_seconds: int = 3600):
    client = await get_redis_client()
    if not client:
        return
    try:
        await client.set(key, json.dumps(value), ex=ttl_seconds)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.debug(f"Cache set failed for {key}: {exc}")
