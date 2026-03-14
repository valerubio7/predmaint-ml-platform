import logging
import os
from pathlib import Path

from prefect import flow, task

from data.loader import load_raw_data
from data.transformer import build_features
from monitoring.drift import detect_drift
from training.train import load_config_task, training_pipeline

logger = logging.getLogger(__name__)

RETRAINING_TRIGGER = os.getenv("RETRAINING_TRIGGER", "true").lower() == "true"
PRODUCTION_SAMPLE_START_ROW = int(os.getenv("PRODUCTION_SAMPLE_START_ROW", "7000"))


@task(name="check-drift")
def check_drift(config: dict) -> bool:
    raw_path = Path(config["data"]["raw_path"])
    df = load_raw_data(raw_path)
    X, _ = build_features(df)
    production_sample = X.iloc[PRODUCTION_SAMPLE_START_ROW:]
    result = detect_drift(production_sample)
    logger.info("Drift detected: %s", result["drift_detected"])
    return result["drift_detected"]


@flow(name="monitoring-pipeline")
def monitoring_pipeline() -> None:
    config = load_config_task()
    drift: bool = check_drift(config)  # type: ignore[assignment]

    if drift and RETRAINING_TRIGGER:
        logger.info("Drift detected — triggering retraining...")
        training_pipeline()
    elif drift:
        logger.info("Drift detected — retraining disabled by RETRAINING_TRIGGER.")
    else:
        logger.info("No drift detected — model is stable.")
