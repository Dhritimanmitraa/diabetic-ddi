"""Latency SLO targets per API route."""
from __future__ import annotations

# Default SLO for endpoints without a specific override.
DEFAULT_LATENCY_TARGET_MS = 500.0

# Exact route targets (FastAPI route paths).
EXACT_ROUTE_TARGETS_MS: dict[str, float] = {
    "/health": 120.0,
    "/health/live": 80.0,
    "/health/ready": 120.0,
    "/metrics": 200.0,
    "/stats": 300.0,
    "/ml/model-info": 600.0,
    "/ml/comparison": 1200.0,
    "/diabetic/risk-check/llm": 5000.0,
    "/diabetic/report/{patient_id}/pdf": 3500.0,
    "/diabetic/analyze-report": 9000.0,
    "/diabetic/analyze-report/personalized-ddi": 6000.0,
    "/prescription/upload": 10000.0,
    "/prescription/upload/base64": 10000.0,
    "/prescription/chat": 7000.0,
}

# Prefix targets cover all remaining endpoints under each group.
PREFIX_ROUTE_TARGETS_MS: tuple[tuple[str, float], ...] = (
    ("/auth", 300.0),
    ("/drugs", 300.0),
    ("/interactions", 450.0),
    ("/alternatives", 700.0),
    ("/ocr", 3000.0),
    ("/history", 350.0),
    ("/ml", 1200.0),
    ("/adherence", 450.0),
    ("/admin", 500.0),
    ("/diabetic", 900.0),
    ("/prescription", 1200.0),
)


def get_latency_target_ms(route_path: str) -> float:
    """Return latency target in milliseconds for a route path."""
    if route_path in EXACT_ROUTE_TARGETS_MS:
        return EXACT_ROUTE_TARGETS_MS[route_path]

    for prefix, target in PREFIX_ROUTE_TARGETS_MS:
        if route_path.startswith(prefix):
            return target

    return DEFAULT_LATENCY_TARGET_MS
