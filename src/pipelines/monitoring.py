import logging
import os
from pathlib import Path

import pandas as pd
from prefect import flow, task

from monitoring.drift.detector import (
    REFERENCE_SPLIT_RATIO,
    _compute_split_index,
    detect_drift,
)
from pipelines.data import load_features
from pipelines.training import load_config_task, training_pipeline

logger = logging.getLogger(__name__)

_BOOL_TRUE = "true"
_BOOL_FALSE = "false"


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized == _BOOL_TRUE:
        return True
    if normalized == _BOOL_FALSE:
        return False
    raise ValueError(
        f"Environment variable '{name}' must be 'true' or 'false', got: {raw!r}"
    )


RETRAINING_TRIGGER: bool = _read_bool_env("RETRAINING_TRIGGER", default=True)


def _build_production_sample(raw_path: Path) -> pd.DataFrame:
    """Return the post-reference portion of the dataset as the production sample."""
    X, _ = load_features(raw_path)
    split = _compute_split_index(len(X), REFERENCE_SPLIT_RATIO)
    return X.iloc[split:]


@task(name="check-drift")
def check_drift(config: dict) -> bool:
    raw_path = Path(config["data"]["raw_path"])
    production_sample = _build_production_sample(raw_path)
    result = detect_drift(production_sample)
    logger.info(
        "Drift check complete — detected: %s, report: %s",
        result["drift_detected"],
        result["report_path"],
    )
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
