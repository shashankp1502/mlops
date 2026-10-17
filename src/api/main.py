"""
FastAPI serving layer for the heart disease classifier.

Design notes:
- The model is loaded exactly once at process startup (`lifespan`), not
  per-request — joblib.load of a ColumnTransformer+LogisticRegression
  pipeline is cheap, but there's no reason to redo it 1000x/sec.
- Every request to /predict gets logged (structured, one line, to stdout)
  with a request id, the outcome, and latency — this satisfies the
  "integrate logging of API requests" requirement and is also just good
  practice for debugging a deployed service.
- /metrics exposes Prometheus counters/histograms so Prometheus can scrape
  this service directly; see k8s/monitoring/ for the scrape config.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.schemas import HealthResponse, PatientFeatures, PredictionResponse
from src.models.predict import DEFAULT_MODEL_PATH, HeartDiseaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart_disease_api")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "method", "status_code"]
)
PREDICTION_COUNT = Counter(
    "model_predictions_total", "Total predictions made, by predicted label", ["label"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)

_model_holder: dict = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _model_holder["model"] = HeartDiseaseModel(DEFAULT_MODEL_PATH)
        logger.info("Model loaded successfully from %s", DEFAULT_MODEL_PATH)
    except FileNotFoundError as exc:
        # don't crash the whole process on import — let /health report the
        # problem clearly instead of the pod silently CrashLoopBackOff-ing
        # with no explanation
        logger.error("Model not loaded at startup: %s", exc)
        _model_holder["model"] = None
    yield
    _model_holder.clear()


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts heart disease risk from patient health data (UCI Cleveland dataset).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_and_time_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    REQUEST_COUNT.labels(
        endpoint=request.url.path, method=request.method, status_code=response.status_code
    ).inc()

    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id, request.method, request.url.path, response.status_code, duration * 1000,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "heart-disease-prediction-api",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    model = _model_holder.get("model")
    champion_name = None
    metadata_path = Path(DEFAULT_MODEL_PATH).parent / "model_metadata.json"
    if metadata_path.exists():
        import json
        champion_name = json.loads(metadata_path.read_text()).get("champion_model")

    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        champion_model=champion_name,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: PatientFeatures):
    model: HeartDiseaseModel | None = _model_holder.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health.")

    result = model.predict_one(payload.model_dump())
    PREDICTION_COUNT.labels(label=result["label"]).inc()

    return PredictionResponse(
        prediction=result["prediction"],
        label=result["label"],
        confidence=round(result["confidence"], 4),
        probability_disease=round(result["probability_disease"], 4),
        model_version=str(DEFAULT_MODEL_PATH.name),
    )


@app.get("/metrics", tags=["meta"])
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
