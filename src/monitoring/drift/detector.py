import logging
import os
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from core.features import build_features
from core.loader import load_raw_data

logger = logging.getLogger(__name__)

REFERENCE_PATH = Path("data/processed/reference.parquet")
REPORTS_PATH = Path("reports/drift")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))

# Fraction of the dataset used as the reference (historical) split.
REFERENCE_SPLIT_RATIO = float(os.getenv("REFERENCE_SPLIT_RATIO", "0.7"))


def build_reference_dataset() -> None:
    """Save the first REFERENCE_SPLIT_RATIO of data as the reference dataset."""
    df = load_raw_data()
    X, y = build_features(df)

    reference = X.copy()
    reference["target"] = y

    split = int(len(reference) * REFERENCE_SPLIT_RATIO)
    reference_data = reference.iloc[:split]

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference_data.to_parquet(REFERENCE_PATH, index=False)
    logger.info("Reference dataset saved: %s", reference_data.shape)


def detect_drift(production_data: pd.DataFrame) -> dict:
    """Compare production data against reference and detect drift."""
    reference = pd.read_parquet(REFERENCE_PATH)

    feature_columns = [c for c in reference.columns if c != "target"]

    report = Report(metrics=[DataDriftPreset()])

    ref_dataset = Dataset.from_pandas(
        reference[feature_columns], data_definition=DataDefinition()
    )
    prod_dataset = Dataset.from_pandas(
        production_data[feature_columns], data_definition=DataDefinition()
    )

    snapshot = report.run(current_data=prod_dataset, reference_data=ref_dataset)

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_PATH / "drift_report.html"
    snapshot.save_html(str(report_path))
    logger.info("Drift report saved: %s", report_path)

    result = snapshot.dict()
    # evidently 0.7+: first metric is DriftedColumnsCount with value.share
    drift_share = result["metrics"][0]["value"]["share"]
    drift_detected = drift_share >= DRIFT_THRESHOLD

    return {
        "drift_detected": drift_detected,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_reference_dataset()

    df = load_raw_data()
    X, y = build_features(df)
    production_sample = X.iloc[int(len(X) * REFERENCE_SPLIT_RATIO) :]

    result = detect_drift(production_sample)
    logger.info("Drift detected: %s", result["drift_detected"])
    logger.info("Report: %s", result["report_path"])
