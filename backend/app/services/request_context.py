"""Request-scoped context helpers for logging and tracing."""
from __future__ import annotations

from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def set_request_context(request_id: str, trace_id: str) -> None:
    """Store request identifiers for downstream log records."""
    request_id_var.set(request_id)
    trace_id_var.set(trace_id)


def clear_request_context() -> None:
    """Reset request-scoped identifiers after a request completes."""
    request_id_var.set("-")
    trace_id_var.set("-")


def get_request_id() -> str:
    """Return the current request id."""
    return request_id_var.get()


def get_trace_id() -> str:
    """Return the current trace id."""
    return trace_id_var.get()
