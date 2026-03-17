"""Application middleware registration."""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.config import get_settings
from app.services.rate_limiter import rate_limit

logger = logging.getLogger(__name__)

RequestHandler = Callable[[Request], Awaitable[Response]]


def setup_middleware(app: FastAPI) -> None:
    """Register HTTP middleware and CORS configuration."""
    settings = get_settings()
    allowed_origins = [
        origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

    @app.middleware("http")
    async def log_drug_requests(request: Request, call_next: RequestHandler) -> Response:
        """Log drug search/list requests for visibility."""
        path = request.url.path
        if path.startswith("/v1/drugs") or path.startswith("/drugs"):
            logger.info(
                "http_request",
                extra={
                    "path": path,
                    "method": request.method,
                    "query": dict(request.query_params),
                    "client_ip": request.client.host if request.client else "unknown",
                },
            )
        return await call_next(request)

    @app.middleware("http")
    async def request_context_middleware(
        request: Request, call_next: RequestHandler
    ) -> Response:
        """Attach request tracing, timing, and global rate limiting."""
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        try:
            await rate_limit(
                request,
                limit=settings.RATE_LIMIT_REQUESTS_PER_MIN,
                window_seconds=60,
                key_prefix="global",
            )
            response = await call_next(request)
        except HTTPException as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail, "request_id": request_id},
            )
        except Exception:
            logger.exception(
                "Unhandled request error",
                extra={"request_id": request_id, "path": request.url.path},
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "request_id": request_id,
                },
            )

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-MS"] = str(duration_ms)

        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "client_ip": request.client.host if request.client else "unknown",
            },
        )
        return response
