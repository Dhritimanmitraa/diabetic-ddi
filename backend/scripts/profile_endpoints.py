"""Simple endpoint profiler for local backend latency baselining.

Run with backend server active:
    python backend/scripts/profile_endpoints.py
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import httpx

BASE_URL = "http://127.0.0.1:8001"


@dataclass
class EndpointSpec:
    method: str
    path: str
    auth: bool = False
    json_body: dict | None = None
    expected_status: tuple[int, ...] = (200,)
    repeats: int = 3


def _request(client: httpx.Client, spec: EndpointSpec, headers: dict[str, str]) -> float:
    started = time.perf_counter()
    response = client.request(
        spec.method,
        spec.path,
        headers=headers if spec.auth else None,
        json=spec.json_body,
        timeout=60.0,
    )
    if response.status_code not in spec.expected_status:
        raise RuntimeError(f"{spec.method} {spec.path} returned {response.status_code}: {response.text[:200]}")
    return (time.perf_counter() - started) * 1000.0


def _summary(label: str, samples: list[float]) -> dict[str, float]:
    sorted_samples = sorted(samples)
    if len(sorted_samples) == 1:
        p95 = sorted_samples[0]
    else:
        # Inclusive percentile interpolation bounded by observed min/max.
        rank = 0.95 * (len(sorted_samples) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(sorted_samples) - 1)
        weight = rank - lower
        p95 = sorted_samples[lower] + (sorted_samples[upper] - sorted_samples[lower]) * weight

    return {
        "label": label,
        "avg_ms": round(statistics.mean(samples), 2),
        "p95_ms": round(p95, 2),
        "max_ms": round(max(samples), 2),
    }


def main() -> None:
    specs = [
        EndpointSpec("GET", "/health"),
        EndpointSpec("GET", "/stats"),
        EndpointSpec("GET", "/drugs/search?query=aspirin&limit=10"),
        EndpointSpec("POST", "/interactions/check", json_body={"drug1_name": "Aspirin", "drug2_name": "Warfarin"}),
        EndpointSpec("POST", "/ml/predict", json_body={"drug1_name": "Aspirin", "drug2_name": "Warfarin"}),
        EndpointSpec("GET", "/prescription/health/status"),
        EndpointSpec("GET", "/prescription/history?limit=10", auth=True),
        EndpointSpec("POST", "/diabetic/risk-check", auth=True, json_body={"patient_id": "LAT-P", "drug_name": "Metformin"}),
        EndpointSpec("POST", "/diabetic/risk-check/llm", auth=True, json_body={"patient_id": "LAT-P", "drug_name": "Metformin"}, repeats=2),
        EndpointSpec("POST", "/diabetic/medication-list-check", auth=True, json_body={"patient_id": "LAT-P"}),
        EndpointSpec("GET", "/diabetic/report/LAT-P", auth=True),
        EndpointSpec("GET", "/diabetic/analyzer-status", repeats=2),
        EndpointSpec("GET", "/diabetic/report/LAT-P/pdf", auth=True, repeats=2),
    ]

    with httpx.Client(base_url=BASE_URL) as client:
        # Ensure auth session for protected endpoints.
        register = client.post(
            "/auth/register",
            json={"username": "latency_user", "email": "latency@example.com", "password": "LatencyPass123!"},
            timeout=10,
        )
        if register.status_code == 201:
            payload = register.json()
        else:
            payload = client.post(
                "/auth/login",
                json={"username": "latency_user", "password": "LatencyPass123!"},
                timeout=10,
            ).json()
        token = payload["access_token"]
        auth_headers = {"Authorization": f"Bearer {token}"}

        # Seed a lightweight diabetic profile.
        client.post(
            "/diabetic/patients",
            headers=auth_headers,
            json={"patient_id": "LAT-P", "name": "Latency Probe", "age": 54, "diabetes_type": "type_2"},
            timeout=10,
        )
        client.post(
            "/diabetic/patients/LAT-P/medications",
            headers=auth_headers,
            json={"drug_name": "Metformin", "dose": "500mg"},
            timeout=10,
        )

        rows: list[dict[str, float]] = []
        for spec in specs:
            samples = [_request(client, spec, auth_headers) for _ in range(spec.repeats)]
            rows.append(_summary(f"{spec.method} {spec.path}", samples))

    rows.sort(key=lambda row: row["avg_ms"], reverse=True)
    print("\nTop 5 slowest routes by average latency:\n")
    for row in rows[:5]:
        print(
            f"- {row['label']}: avg={row['avg_ms']}ms, p95={row['p95_ms']}ms, max={row['max_ms']}ms"
        )

    print("\nFull profile:\n")
    for row in rows:
        print(
            f"{row['label']:<55} avg={row['avg_ms']:>8}ms  p95={row['p95_ms']:>8}ms  max={row['max_ms']:>8}ms"
        )


if __name__ == "__main__":
    main()
