"""
Robust API Client with rate limiting, retries, and circuit breaker pattern.

This module provides a production-grade HTTP client for fetching
drug data from multiple APIs with intelligent failure handling.
"""
import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
import random

import aiohttp

logger = logging.getLogger(__name__)


# Rotating User-Agent pool to avoid blocking
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting per API."""
    requests_per_minute: int
    burst_limit: int = 5
    
    
@dataclass
class CircuitBreakerState:
    """State for circuit breaker pattern."""
    failures: int = 0
    last_failure: Optional[datetime] = None
    is_open: bool = False
    open_until: Optional[datetime] = None


@dataclass
class CacheEntry:
    """Cache entry with TTL."""
    data: Any
    created_at: datetime
    ttl_seconds: int = 3600  # 1 hour default
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)


class TokenBucketRateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, tokens_per_second: float, bucket_size: int):
        self.tokens_per_second = tokens_per_second
        self.bucket_size = bucket_size
        self.tokens = bucket_size
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.
        Returns the time waited.
        """
        async with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            
            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.tokens_per_second
            
            await asyncio.sleep(wait_time)
            self._refill()
            self.tokens -= tokens
            return wait_time
    
    def _refill(self):
        """Refill tokens based on time elapsed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.tokens_per_second
        self.tokens = min(self.bucket_size, self.tokens + new_tokens)
        self.last_refill = now


class RobustAPIClient:
    """
    Production-grade API client with:
    - Rate limiting (token bucket)
    - Exponential backoff retries
    - Circuit breaker pattern
    - Response caching
    - Rotating User-Agents
    """
    
    # API rate limits (requests per minute)
    RATE_LIMITS: Dict[str, RateLimitConfig] = {
        "openfda": RateLimitConfig(requests_per_minute=240, burst_limit=10),
        "rxnorm": RateLimitConfig(requests_per_minute=1200, burst_limit=20),  # 20/sec
        "dailymed": RateLimitConfig(requests_per_minute=100, burst_limit=5),
        "pubchem": RateLimitConfig(requests_per_minute=300, burst_limit=10),
        "drugbank": RateLimitConfig(requests_per_minute=60, burst_limit=3),
        "default": RateLimitConfig(requests_per_minute=60, burst_limit=5),
    }
    
    # Circuit breaker thresholds
    CIRCUIT_FAILURE_THRESHOLD = 5
    CIRCUIT_RECOVERY_TIME = timedelta(minutes=5)
    
    def __init__(self, cache_ttl: int = 3600):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._rate_limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._circuit_breakers: Dict[str, CircuitBreakerState] = defaultdict(CircuitBreakerState)
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"Accept": "application/json"}
            )
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _get_rate_limiter(self, source: str) -> TokenBucketRateLimiter:
        """Get or create rate limiter for a source."""
        if source not in self._rate_limiters:
            config = self.RATE_LIMITS.get(source, self.RATE_LIMITS["default"])
            tokens_per_second = config.requests_per_minute / 60.0
            self._rate_limiters[source] = TokenBucketRateLimiter(
                tokens_per_second=tokens_per_second,
                bucket_size=config.burst_limit
            )
        return self._rate_limiters[source]
    
    def _get_cache_key(self, url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from URL and params."""
        key_str = url + (str(sorted(params.items())) if params else "")
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if not expired."""
        entry = self._cache.get(cache_key)
        if entry and not entry.is_expired:
            logger.debug(f"Cache hit for {cache_key[:8]}...")
            return entry.data
        return None
    
    def _set_cache(self, cache_key: str, data: Any, ttl: Optional[int] = None):
        """Set data in cache."""
        self._cache[cache_key] = CacheEntry(
            data=data,
            created_at=datetime.now(),
            ttl_seconds=ttl or self.cache_ttl
        )
    
    def _is_circuit_open(self, source: str) -> bool:
        """Check if circuit breaker is open for a source."""
        state = self._circuit_breakers[source]
        
        if state.is_open:
            if datetime.now() > state.open_until:
                # Try to recover
                logger.info(f"Circuit breaker for {source} attempting recovery")
                state.is_open = False
                state.failures = 0
                return False
            return True
        return False
    
    def _record_failure(self, source: str):
        """Record a failure for circuit breaker."""
        state = self._circuit_breakers[source]
        state.failures += 1
        state.last_failure = datetime.now()
        
        if state.failures >= self.CIRCUIT_FAILURE_THRESHOLD:
            state.is_open = True
            state.open_until = datetime.now() + self.CIRCUIT_RECOVERY_TIME
            logger.warning(f"Circuit breaker OPEN for {source} until {state.open_until}")
    
    def _record_success(self, source: str):
        """Record a success, reset circuit breaker."""
        state = self._circuit_breakers[source]
        state.failures = 0
        state.is_open = False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with rotating User-Agent."""
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "application/json,text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
    
    async def fetch(
        self,
        url: str,
        source: str = "default",
        params: Optional[Dict] = None,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Fetch data from URL with all reliability features.
        
        Args:
            url: URL to fetch
            source: API source name (for rate limiting)
            params: Query parameters
            max_retries: Maximum retry attempts
            backoff_base: Base delay for exponential backoff
            use_cache: Whether to use caching
            cache_ttl: Custom cache TTL in seconds
            
        Returns:
            JSON response or None if all attempts fail
        """
        # Check cache first
        cache_key = self._get_cache_key(url, params)
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Check circuit breaker
        if self._is_circuit_open(source):
            logger.warning(f"Circuit breaker open for {source}, skipping request")
            return None
        
        # Rate limiting
        rate_limiter = self._get_rate_limiter(source)
        wait_time = await rate_limiter.acquire()
        if wait_time > 0:
            logger.debug(f"Rate limited: waited {wait_time:.2f}s for {source}")
        
        # Retry loop with exponential backoff
        session = await self._get_session()
        last_error = None
        
        for attempt in range(max_retries):
            try:
                headers = self._get_headers()
                
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._record_success(source)
                        
                        # Cache successful response
                        if use_cache:
                            self._set_cache(cache_key, data, cache_ttl)
                        
                        logger.info(f"✓ {source}: Fetched {url[:50]}...")
                        return data
                    
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited by {source}, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    elif response.status >= 500:  # Server error
                        logger.warning(f"{source} server error: {response.status}")
                        last_error = f"HTTP {response.status}"
                    
                    else:
                        logger.warning(f"{source} returned {response.status} for {url}")
                        last_error = f"HTTP {response.status}"
                        
            except asyncio.TimeoutError:
                last_error = "Timeout"
                logger.warning(f"{source} timeout (attempt {attempt + 1}/{max_retries})")
                
            except aiohttp.ClientError as e:
                last_error = str(e)
                logger.warning(f"{source} client error: {e}")
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"{source} unexpected error: {e}")
            
            # Exponential backoff
            if attempt < max_retries - 1:
                delay = backoff_base * (2 ** attempt) + random.uniform(0, 1)
                logger.debug(f"Retrying {source} in {delay:.2f}s...")
                await asyncio.sleep(delay)
        
        # All retries failed
        self._record_failure(source)
        logger.error(f"✗ {source}: All {max_retries} attempts failed. Last error: {last_error}")
        return None
    
    async def fetch_with_fallback(
        self,
        requests: List[Dict[str, Any]],
        stop_on_success: bool = True,
    ) -> Optional[Dict]:
        """
        Try multiple API sources in order until one succeeds.
        
        Args:
            requests: List of request configs, each with 'url', 'source', 'params'
            stop_on_success: If True, stop after first successful response
            
        Returns:
            First successful response or None
        """
        for req in requests:
            result = await self.fetch(
                url=req["url"],
                source=req.get("source", "default"),
                params=req.get("params"),
                max_retries=req.get("max_retries", 2),  # Fewer retries for fallback chain
            )
            
            if result is not None:
                if stop_on_success:
                    return result
                    
        return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all API sources."""
        status = {}
        
        for source, state in self._circuit_breakers.items():
            status[source] = {
                "healthy": not state.is_open,
                "failures": state.failures,
                "circuit_open": state.is_open,
                "open_until": state.open_until.isoformat() if state.open_until else None,
            }
        
        return {
            "apis": status,
            "cache_size": len(self._cache),
            "timestamp": datetime.now().isoformat(),
        }


# Singleton instance
_client: Optional[RobustAPIClient] = None


def get_api_client() -> RobustAPIClient:
    """Get or create the singleton API client."""
    global _client
    if _client is None:
        _client = RobustAPIClient()
    return _client


async def close_api_client():
    """Close the singleton API client."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
