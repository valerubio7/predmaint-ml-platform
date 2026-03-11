import os
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from data.loader import load_raw_data
from data.transformer import build_features

REFERENCE_PATH = Path("data/processed/reference.parquet")
REPORTS_PATH = Path("reports/drift")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))


def build_reference_dataset() -> None:
    """Save first 70% of data as reference (simulates historical training data)."""
    df = load_raw_data()
    X, y = build_features(df)

    reference = X.copy()
    reference["target"] = y

    split = int(len(reference) * 0.7)
    reference_data = reference.iloc[:split]

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference_data.to_parquet(REFERENCE_PATH, index=False)
    print(f"Reference dataset saved: {reference_data.shape}")


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
    print(f"Drift report saved: {report_path}")

    result = snapshot.dict()
    # evidently 0.7+: first metric is DriftedColumnsCount with value.share
    drift_share = result["metrics"][0]["value"]["share"]
    drift_detected = drift_share >= DRIFT_THRESHOLD

    return {
        "drift_detected": drift_detected,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    build_reference_dataset()

    df = load_raw_data()
    X, y = build_features(df)
    production_sample = X.iloc[7000:]

    result = detect_drift(production_sample)
    print(f"Drift detected: {result['drift_detected']}")
    print(f"Report: {result['report_path']}")
