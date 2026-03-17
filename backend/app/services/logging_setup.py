"""Structured logging configuration with request context support."""
from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger

from app.config import get_settings
from app.services.request_context import get_request_id, get_trace_id


class RequestContextFilter(logging.Filter):
    """Attach request identifiers to every emitted log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(record, "request_id", get_request_id())
        record.trace_id = getattr(record, "trace_id", get_trace_id())
        return True


def configure_logging(level: int | None = None) -> None:
    """Configure root logging as structured JSON."""
    settings = get_settings()
    resolved_level = level if level is not None else getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestContextFilter())
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(trace_id)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    )

    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(resolved_level)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
