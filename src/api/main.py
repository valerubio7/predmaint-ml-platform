import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
API_ENV = os.getenv("API_ENV", "development")

app = FastAPI(
    title="PredMaint API",
    description="Predictive maintenance inference API",
    version="0.1.0",
)

_model: Pipeline | None = None


def get_model() -> Pipeline:
    """Load the model lazily; raises 503 if the file is not yet present."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model not found at {MODEL_PATH}. "
                    "Train the model and copy model.pkl into the models/ directory."
                ),
            )
        _model = joblib.load(MODEL_PATH)
    if _model is None:
        raise RuntimeError(f"Model loaded from {MODEL_PATH} is None")
    return _model


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

        return PredictionResponse(
            failure_predicted=bool(prediction),
            failure_probability=round(float(probability), 4),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
