import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from core.features import build_features
from core.loader import load_raw_data

logger = logging.getLogger(__name__)


def _read_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Environment variable '{name}' must be a float, got: {raw!r}")


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

REFERENCE_PATH = _PROJECT_ROOT / "data" / "processed" / "reference.parquet"
REPORTS_PATH = _PROJECT_ROOT / "reports" / "drift"

TARGET_COLUMN = "target"
DRIFT_METRIC_NAME_PREFIX = "DriftedColumnsCount"

DRIFT_THRESHOLD: float = _read_float_env("DRIFT_THRESHOLD", 0.05)
REFERENCE_SPLIT_RATIO: float = _read_float_env("REFERENCE_SPLIT_RATIO", 0.7)


@dataclass(frozen=True)
class ReferenceDatasetInfo:
    path: Path
    row_count: int
    columns: list


def _compute_split_index(total_rows: int, ratio: float) -> int:
    return int(total_rows * ratio)


def build_reference_dataset() -> ReferenceDatasetInfo:
    """Save the first REFERENCE_SPLIT_RATIO of data as the reference dataset."""
    df = load_raw_data()
    X, y = build_features(df)

    reference = X.copy()
    reference[TARGET_COLUMN] = y

    split = _compute_split_index(len(reference), REFERENCE_SPLIT_RATIO)
    reference_data = reference.iloc[:split]

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference_data.to_parquet(REFERENCE_PATH, index=False)

    info = ReferenceDatasetInfo(
        path=REFERENCE_PATH,
        row_count=len(reference_data),
        columns=list(reference_data.columns),
    )
    logger.info("Reference dataset saved: path=%s rows=%d", info.path, info.row_count)
    return info


def _load_reference(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at '{path}'. "
            "Run build_reference_dataset() first."
        )
    return pd.read_parquet(path)


def _validate_production_columns(
    production_data: pd.DataFrame, expected_columns: list
) -> None:
    missing = set(expected_columns) - set(production_data.columns)
    if missing:
        raise ValueError(
            f"production_data is missing columns required by the reference: {missing}"
        )


def _run_drift_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    feature_columns: list,
) -> Any:
    report = Report(metrics=[DataDriftPreset()])
    ref_dataset = Dataset.from_pandas(
        reference[feature_columns], data_definition=DataDefinition()
    )
    prod_dataset = Dataset.from_pandas(
        production[feature_columns], data_definition=DataDefinition()
    )
    return report.run(current_data=prod_dataset, reference_data=ref_dataset)


def _extract_drift_share(snapshot_result: dict) -> float:
    metrics = snapshot_result.get("metrics", [])
    for metric in metrics:
        metric_name = metric.get("metric_name", "")
        if metric_name.startswith(DRIFT_METRIC_NAME_PREFIX):
            return metric["value"]["share"]
    raise ValueError(
        f"Expected metric starting with '{DRIFT_METRIC_NAME_PREFIX}' not found in "
        f"Evidently snapshot. Available metrics: "
        f"{[m.get('metric_name') for m in metrics]}"
    )


def _save_report(snapshot: Any, base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = base_dir / f"drift_report_{timestamp}.html"
    snapshot.save_html(str(report_path))
    logger.info("Drift report saved: %s", report_path)
    return report_path


def _decide_drift(drift_share: float, threshold: float) -> bool:
    return drift_share >= threshold


def detect_drift(production_data: pd.DataFrame) -> dict:
    """Compare production data against reference and detect drift."""
    reference = _load_reference(REFERENCE_PATH)
    feature_columns = [c for c in reference.columns if c != TARGET_COLUMN]

    _validate_production_columns(production_data, feature_columns)

    snapshot = _run_drift_report(reference, production_data, feature_columns)
    report_path = _save_report(snapshot, REPORTS_PATH)

    drift_share = _extract_drift_share(snapshot.dict())
    drift_detected = _decide_drift(drift_share, DRIFT_THRESHOLD)

    return {
        "drift_detected": drift_detected,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_reference_dataset()

    df = load_raw_data()
    X, _ = build_features(df)
    split = _compute_split_index(len(X), REFERENCE_SPLIT_RATIO)
    production_sample = X.iloc[split:]

    result = detect_drift(production_sample)
    logger.info("Drift detected: %s", result["drift_detected"])
    logger.info("Report: %s", result["report_path"])
