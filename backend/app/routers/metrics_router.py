"""Prometheus metrics endpoint."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.services.observability import render_prometheus_metrics

router = APIRouter(tags=["Statistics"])


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Expose Prometheus-compatible application metrics."""
    return PlainTextResponse(
        render_prometheus_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
