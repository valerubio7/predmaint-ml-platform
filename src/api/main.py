import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
import joblib
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from monitoring.metrics.prometheus import MODEL_CONFIDENCE_SCORE, PREDICTIONS_TOTAL

logger = logging.getLogger(__name__)

_raw_model_path = os.getenv("MODEL_PATH", "models/model.pkl")
# Normalize s3:/ → s3:// in case SSM value was stored with a single slash
MODEL_PATH = (
    "s3://" + _raw_model_path[len("s3:/") :]
    if _raw_model_path.startswith("s3:/") and not _raw_model_path.startswith("s3://")
    else _raw_model_path
)
API_ENV = os.getenv("API_ENV", "development")

_model = None
_LOCAL_MODEL_CACHE = Path("/tmp/model.pkl")  # nosec B108 — standard cache dir in containers


def _download_from_s3(s3_uri: str) -> Path:
    """Download a model from an S3 URI (s3://bucket/key) to a local cache path."""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {s3_uri}")

    uri = s3_uri[len("s3://") :]
    bucket, _, key = uri.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    logger.info("Downloading model from %s", s3_uri)
    try:
        s3 = boto3.client("s3")
        _LOCAL_MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(_LOCAL_MODEL_CACHE))
        logger.info("Model downloaded to %s", _LOCAL_MODEL_CACHE)
    except (BotoCoreError, ClientError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to download model from {s3_uri}: {e}",
        )
    return _LOCAL_MODEL_CACHE


def _resolve_model_path() -> Path:
    """Return a local Path to the model, downloading from S3 if needed."""
    if MODEL_PATH.startswith("s3://"):
        return _download_from_s3(MODEL_PATH)

    local = Path(MODEL_PATH)
    if not local.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not found at {local}. "
                "Train the model or set MODEL_PATH to an s3:// URI."
            ),
        )
    return local


def get_model():
    """Load the model lazily, downloading from S3 when MODEL_PATH is an s3:// URI."""
    global _model
    if _model is None:
        path = _resolve_model_path()
        _model = joblib.load(path)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Pre-load the model at startup so the first request isn't slow.

    Logs a warning if the model is unavailable rather than crashing the server —
    /health will still respond while /predict returns 503 until the model is ready.
    """
    try:
        get_model()
        logger.info("Model loaded successfully (env=%s)", API_ENV)
    except HTTPException as e:
        logger.warning("Model not available at startup: %s", e.detail)
    yield


app = FastAPI(
    title="PredMaint API",
    description="Predictive maintenance inference API",
    version="0.1.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


class SensorData(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int
    type_h: int
    type_l: int
    type_m: int


class PredictionResponse(BaseModel):
    failure_predicted: bool
    failure_probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: SensorData):
    model = get_model()
    try:
        features = pd.DataFrame(
            [
                {
                    "Air_temperature_K": data.air_temperature,
                    "Process_temperature_K": data.process_temperature,
                    "Rotational_speed_rpm": data.rotational_speed,
                    "Torque_Nm": data.torque,
                    "Tool_wear_min": data.tool_wear,
                    "Type_H": data.type_h,
                    "Type_L": data.type_l,
                    "Type_M": data.type_m,
                }
            ]
        )

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        outcome = "failure" if bool(prediction) else "normal"
        PREDICTIONS_TOTAL.labels(outcome=outcome).inc()
        MODEL_CONFIDENCE_SCORE.observe(float(probability))

        return PredictionResponse(
            failure_predicted=bool(prediction),
            failure_probability=round(float(probability), 4),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
