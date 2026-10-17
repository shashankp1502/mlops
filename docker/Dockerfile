# syntax=docker/dockerfile:1

# ---- Build context note -------------------------------------------------
# This Dockerfile is meant to be built from the *repo root*, not from
# inside docker/, since it needs both src/ and the already-trained
# models/model.pkl:
#
#   docker build -t heart-disease-api:latest -f docker/Dockerfile .
#
# models/model.pkl must exist before building — run
# `python -m src.models.train` locally first (or let CI's train job
# produce it, as the ci.yml docker job does).
# --------------------------------------------------------------------------

# ---- Stage 1: build a venv with all deps -----------------------------
# Keeps the compiler toolchain (needed for a couple of scientific wheels
# on some architectures) out of the final image entirely.
FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

# ---- Stage 2: slim runtime image --------------------------------------
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src/ src/
COPY models/model.pkl models/model.pkl
COPY models/model_metadata.json models/model_metadata.json

# run as a non-root user — not strictly required for a local/CI demo, but
# it's a one-line change and matches what you'd actually want in a real
# deployment
RUN useradd --create-home appuser
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
