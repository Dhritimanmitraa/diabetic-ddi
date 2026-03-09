# ── Stage 1: Build ───────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for building native wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Runtime system deps (Tesseract for OCR, libgomp for LightGBM)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgomp1 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create directories for data and models
RUN mkdir -p /app/models /app/data /app/logs /app/chroma_db

# Non-root user for security
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
