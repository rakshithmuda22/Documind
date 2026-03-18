# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download the embedding model at build time (~80 MB)
# so the first request does not require a download
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy pre-downloaded model cache
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application source
COPY . .

# Create a non-root user and transfer ownership
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app \
    && mkdir -p /home/appuser/.cache \
    && cp -r /root/.cache/huggingface /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /home/appuser/.cache

USER appuser

EXPOSE 7860

# Single worker required — in-memory state is not shared across workers
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1
