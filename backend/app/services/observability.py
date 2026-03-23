"""Lightweight observability helpers for metrics and trace IDs."""
from __future__ import annotations

from collections import defaultdict
from time import time
from threading import Lock

_lock = Lock()
_request_totals: dict[tuple[str, str, int], int] = defaultdict(int)
_request_duration_sum: dict[tuple[str, str], float] = defaultdict(float)
_request_duration_count: dict[tuple[str, str], int] = defaultdict(int)
_request_duration_buckets: dict[tuple[str, str, float], int] = defaultdict(int)
_slow_request_totals: dict[tuple[str, str], int] = defaultdict(int)
_in_progress = 0
_process_start_time = time()
_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def inc_in_progress() -> None:
    global _in_progress
    with _lock:
        _in_progress += 1


def dec_in_progress() -> None:
    global _in_progress
    with _lock:
        _in_progress = max(0, _in_progress - 1)


def record_request(method: str, path: str, status_code: int, duration_seconds: float) -> None:
    """Record request metrics."""
    labels = (method, path)
    with _lock:
        _request_totals[(method, path, status_code)] += 1
        _request_duration_sum[labels] += duration_seconds
        _request_duration_count[labels] += 1
        for bucket in _BUCKETS:
            if duration_seconds <= bucket:
                _request_duration_buckets[(method, path, bucket)] += 1


def record_slow_request(method: str, path: str) -> None:
    """Record a request that exceeded its latency target."""
    with _lock:
        _slow_request_totals[(method, path)] += 1


def render_prometheus_metrics() -> str:
    """Render metrics in Prometheus exposition format."""
    lines = [
        "# HELP process_start_time_seconds Start time of the process since unix epoch",
        "# TYPE process_start_time_seconds gauge",
        f"process_start_time_seconds {_process_start_time}",
        "# HELP http_requests_total Total HTTP requests",
        "# TYPE http_requests_total counter",
    ]
    with _lock:
        for (method, path, status_code), count in sorted(_request_totals.items()):
            lines.append(
                f'http_requests_total{{method="{method}",path="{path}",status="{status_code}"}} {count}'
            )

        lines.extend(
            [
                "# HELP http_slow_requests_total Requests that exceeded route latency targets",
                "# TYPE http_slow_requests_total counter",
            ]
        )
        for (method, path), count in sorted(_slow_request_totals.items()):
            lines.append(
                f'http_slow_requests_total{{method="{method}",path="{path}"}} {count}'
            )

        lines.extend(
            [
                "# HELP http_requests_in_progress In-flight HTTP requests",
                "# TYPE http_requests_in_progress gauge",
                f"http_requests_in_progress {_in_progress}",
                "# HELP http_request_duration_seconds Request duration histogram",
                "# TYPE http_request_duration_seconds histogram",
            ]
        )

        for (method, path), count in sorted(_request_duration_count.items()):
            cumulative = 0
            for bucket in _BUCKETS:
                cumulative += _request_duration_buckets.get((method, path, bucket), 0)
                lines.append(
                    f'http_request_duration_seconds_bucket{{method="{method}",path="{path}",le="{bucket}"}} {cumulative}'
                )
            lines.append(
                f'http_request_duration_seconds_bucket{{method="{method}",path="{path}",le="+Inf"}} {count}'
            )
            lines.append(
                f'http_request_duration_seconds_sum{{method="{method}",path="{path}"}} {_request_duration_sum[(method, path)]}'
            )
            lines.append(
                f'http_request_duration_seconds_count{{method="{method}",path="{path}"}} {count}'
            )

    return "\n".join(lines) + "\n"
