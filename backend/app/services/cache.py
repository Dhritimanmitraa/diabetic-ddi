"""
Redis cache utilities.

Uses async redis client when REDIS_URL is configured. Falls back gracefully if
Redis is unavailable.

Example usage:
    # Direct cache operations
    from app.services.cache import cache_get_json, cache_set_json
    
    cached = await cache_get_json("drug_search:aspirin")
    if cached is None:
        result = await search_drugs("aspirin")
        await cache_set_json("drug_search:aspirin", result, ttl_seconds=3600)
    
    # Using the caching decorator
    @cached_endpoint(prefix="drug_search", ttl=3600)
    async def search_drugs(query: str, limit: int = 10):
        ...
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from typing import Any, Callable, Optional, TypeVar, ParamSpec

import redis.asyncio as redis  # type: ignore

logger = logging.getLogger(__name__)

_redis_client: Optional[redis.Redis] = None

# Type variables for generic decorator
P = ParamSpec('P')
T = TypeVar('T')


# =============================================================================
# Core Cache Functions
# =============================================================================

async def get_redis_client() -> Optional[redis.Redis]:
    """
    Return a shared async Redis client if REDIS_URL is set and reachable.
    
    Re-validates the connection on each call and reconnects if stale.
    
    Returns:
        Redis client instance or None if unavailable
    """
    global _redis_client
    
    # If we have an existing client, verify it's still alive
    if _redis_client:
        try:
            await _redis_client.ping()
            return _redis_client
        except Exception:
            logger.warning("Redis connection lost, attempting reconnect...")
            try:
                await _redis_client.close()
            except Exception:
                pass
            _redis_client = None

    from app.config import get_settings
    redis_url = get_settings().REDIS_URL
    try:
        _redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        # quick ping to validate connection
        await _redis_client.ping()
        logger.info("Connected to Redis cache")
        return _redis_client
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning(f"Redis not available ({exc}); caching disabled")
        _redis_client = None
        return None


async def cache_get_json(key: str) -> Optional[Any]:
    """
    Get JSON value from cache.
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None if not found/error
    """
    client = await get_redis_client()
    if not client:
        return None
    try:
        data = await client.get(key)
        return json.loads(data) if data else None
    except Exception as exc:  # pragma: no cover - network dependent
        logger.debug(f"Cache get failed for {key}: {exc}")
        return None


async def cache_set_json(key: str, value: Any, ttl_seconds: int = 3600) -> bool:
    """
    Set JSON value in cache.
    
    Args:
        key: Cache key
        value: Value to cache (must be JSON serializable)
        ttl_seconds: Time to live in seconds (default: 1 hour)
        
    Returns:
        True if cached successfully, False otherwise
    """
    client = await get_redis_client()
    if not client:
        return False
    try:
        await client.set(key, json.dumps(value), ex=ttl_seconds)
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        logger.debug(f"Cache set failed for {key}: {exc}")
        return False


async def cache_delete(key: str) -> bool:
    """
    Delete a key from cache.
    
    Args:
        key: Cache key to delete
        
    Returns:
        True if deleted, False otherwise
    """
    client = await get_redis_client()
    if not client:
        return False
    try:
        await client.delete(key)
        return True
    except Exception as exc:
        logger.debug(f"Cache delete failed for {key}: {exc}")
        return False


async def cache_invalidate_pattern(pattern: str) -> int:
    """
    Delete all keys matching a pattern.
    
    Args:
        pattern: Redis pattern (e.g., "drug_search:*")
        
    Returns:
        Number of keys deleted
    """
    client = await get_redis_client()
    if not client:
        return 0
    try:
        keys = await client.keys(pattern)
        if keys:
            return await client.delete(*keys)
        return 0
    except Exception as exc:
        logger.debug(f"Cache invalidate failed for pattern {pattern}: {exc}")
        return 0


# =============================================================================
# Cache Key Helpers
# =============================================================================

def make_cache_key(prefix: str, *args: Any, **kwargs: Any) -> str:
    """
    Generate a consistent cache key from prefix and arguments.
    
    Args:
        prefix: Key prefix (e.g., "drug_search")
        *args: Positional arguments to include in key
        **kwargs: Keyword arguments to include in key
        
    Returns:
        Cache key string
        
    Example:
        >>> make_cache_key("drug_search", "aspirin", limit=10)
        "drug_search:aspirin:limit=10"
    """
    parts = [prefix]
    
    # Add positional args
    for arg in args:
        if arg is not None:
            parts.append(str(arg).lower())
    
    # Add keyword args (sorted for consistency)
    for key in sorted(kwargs.keys()):
        value = kwargs[key]
        if value is not None:
            parts.append(f"{key}={value}")
    
    return ":".join(parts)


def make_cache_key_hash(prefix: str, *args: Any, **kwargs: Any) -> str:
    """
    Generate a hashed cache key for long/complex arguments.
    
    Useful when arguments might contain special characters or be very long.
    
    Args:
        prefix: Key prefix
        *args: Arguments to hash
        **kwargs: Keyword arguments to hash
        
    Returns:
        Cache key with hashed content
    """
    content = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    hash_value = hashlib.md5(content.encode()).hexdigest()[:12]
    return f"{prefix}:{hash_value}"


# =============================================================================
# Caching Decorator
# =============================================================================

def cached_endpoint(
    prefix: str,
    ttl: int = 3600,
    key_builder: Optional[Callable[..., str]] = None,
):
    """
    Decorator to cache async endpoint responses.
    
    Args:
        prefix: Cache key prefix (e.g., "drug_search")
        ttl: Time to live in seconds (default: 1 hour)
        key_builder: Optional custom function to build cache key.
                    If not provided, uses make_cache_key.
                    
    Returns:
        Decorated function with caching
        
    Example:
        @cached_endpoint(prefix="drug_search", ttl=3600)
        async def search_drugs(query: str, limit: int = 10):
            # This result will be cached
            return await db.execute(...)
            
        # Custom key builder
        @cached_endpoint(
            prefix="ml_model_info",
            ttl=86400,  # 24 hours
            key_builder=lambda: "ml_model_info:singleton"
        )
        async def get_model_info():
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = make_cache_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached = await cache_get_json(cache_key)
            if cached is not None:
                logger.debug(f"Cache HIT: {cache_key}")
                return cached
            
            # Execute function
            logger.debug(f"Cache MISS: {cache_key}")
            result = await func(*args, **kwargs)
            
            # Cache the result
            await cache_set_json(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# =============================================================================
# Specialized Cache Functions
# =============================================================================

async def cache_drug_search(query: str, limit: int, results: list) -> bool:
    """
    Cache drug search results.
    
    Args:
        query: Search query
        limit: Result limit
        results: List of drug results (must be serializable)
        
    Returns:
        True if cached successfully
    """
    key = make_cache_key("drug_search", query, limit=limit)
    return await cache_set_json(key, results, ttl_seconds=3600)


async def get_cached_drug_search(query: str, limit: int) -> Optional[list]:
    """
    Get cached drug search results.
    
    Args:
        query: Search query
        limit: Result limit
        
    Returns:
        Cached results or None
    """
    key = make_cache_key("drug_search", query, limit=limit)
    return await cache_get_json(key)


async def cache_model_info(info: dict) -> bool:
    """
    Cache ML model information (long TTL since it rarely changes).
    
    Args:
        info: Model info dictionary
        
    Returns:
        True if cached successfully
    """
    return await cache_set_json("ml_model_info:current", info, ttl_seconds=86400)  # 24 hours


async def get_cached_model_info() -> Optional[dict]:
    """
    Get cached ML model information.
    
    Returns:
        Cached model info or None
    """
    return await cache_get_json("ml_model_info:current")


async def cache_db_stats(stats: dict) -> bool:
    """
    Cache database statistics.
    
    Args:
        stats: Statistics dictionary
        
    Returns:
        True if cached successfully
    """
    return await cache_set_json("db_stats:current", stats, ttl_seconds=300)  # 5 minutes


async def get_cached_db_stats() -> Optional[dict]:
    """
    Get cached database statistics.
    
    Returns:
        Cached stats or None
    """
    return await cache_get_json("db_stats:current")
