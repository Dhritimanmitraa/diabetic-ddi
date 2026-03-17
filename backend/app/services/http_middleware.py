"""HTTP middleware setup with tracing, metrics, and structured request logging."""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.config import get_settings
from app.services.observability import dec_in_progress, inc_in_progress, record_request
from app.services.request_context import clear_request_context, set_request_context
from app.services.rate_limiter import rate_limit

logger = logging.getLogger(__name__)

RequestHandler = Callable[[Request], Awaitable[Response]]


def _extract_trace_id(request: Request) -> str:
    traceparent = request.headers.get("traceparent", "")
    parts = traceparent.split("-")
    if len(parts) >= 4 and len(parts[1]) == 32:
        return parts[1]
    return request.headers.get("X-Request-ID") or uuid.uuid4().hex


def setup_http_middleware(app: FastAPI) -> None:
    """Register CORS, rate limiting, tracing, and request metrics middleware."""
    settings = get_settings()
    allowed_origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "traceparent"],
    )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next: RequestHandler) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        trace_id = _extract_trace_id(request)
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        set_request_context(request_id, trace_id)
        start = time.perf_counter()
        inc_in_progress()

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
                content={"detail": exc.detail, "request_id": request_id, "trace_id": trace_id},
            )
        except Exception:
            logger.exception(
                "Unhandled request error",
                extra={"request_id": request_id, "trace_id": trace_id, "path": request.url.path},
            )
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id, "trace_id": trace_id},
            )
        finally:
            dec_in_progress()

        duration_seconds = time.perf_counter() - start
        duration_ms = round(duration_seconds * 1000, 2)
        route = request.scope.get("route")
        metrics_path = getattr(route, "path", request.url.path)
        record_request(request.method, metrics_path, response.status_code, duration_seconds)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Process-Time-MS"] = str(duration_ms)

        try:
            logger.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "path": request.url.path,
                    "route_path": metrics_path,
                    "method": request.method,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "client_ip": request.client.host if request.client else "unknown",
                },
            )
        finally:
            clear_request_context()
        return response
