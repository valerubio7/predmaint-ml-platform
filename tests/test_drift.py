"""Tests for src/monitoring/drift.py.

Evidently is an external, heavy dependency that writes HTML files and parquet.
All tests use mocking / real lightweight DataFrames to stay fast and isolated.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
    "Type_H",
    "Type_L",
    "Type_M",
]


def _make_feature_df(rows: int = 100) -> pd.DataFrame:
    """Return a synthetic feature DataFrame matching the real schema."""
    import numpy as np

    rng = np.random.default_rng(42)
    data = {
        "Air_temperature_K": rng.normal(300, 2, rows),
        "Process_temperature_K": rng.normal(310, 2, rows),
        "Rotational_speed_rpm": rng.integers(1400, 1700, rows).astype(float),
        "Torque_Nm": rng.normal(40, 5, rows),
        "Tool_wear_min": rng.integers(0, 200, rows).astype(float),
        "Type_H": rng.integers(0, 2, rows),
        "Type_L": rng.integers(0, 2, rows),
        "Type_M": rng.integers(0, 2, rows),
    }
    return pd.DataFrame(data)


def _make_reference_parquet(tmp_path: Path, rows: int = 100) -> Path:
    """Write a reference.parquet to a tmp directory."""
    ref = _make_feature_df(rows)
    ref["target"] = 0
    parquet_path = tmp_path / "reference.parquet"
    ref.to_parquet(parquet_path, index=False)
    return parquet_path


# ---------------------------------------------------------------------------
# build_reference_dataset tests
# ---------------------------------------------------------------------------


def test_build_reference_dataset_creates_parquet(tmp_path, monkeypatch):
    """build_reference_dataset must write a Parquet file."""
    import monitoring.drift as drift_module

    # Patch the path so it writes to tmp_path
    monkeypatch.setattr(drift_module, "REFERENCE_PATH", tmp_path / "reference.parquet")

    # Patch load_raw_data and build_features to avoid real file I/O
    raw_df = pd.DataFrame({"dummy": range(100)})
    feature_df = _make_feature_df(100)
    target_series = pd.Series([0] * 100)

    with (
        patch("monitoring.drift.load_raw_data", return_value=raw_df),
        patch(
            "monitoring.drift.build_features", return_value=(feature_df, target_series)
        ),
    ):
        drift_module.build_reference_dataset()

    assert (tmp_path / "reference.parquet").exists()


def test_build_reference_dataset_saves_70_percent(tmp_path, monkeypatch):
    """Reference dataset must be the first 70% of rows."""
    import monitoring.drift as drift_module

    n = 100
    monkeypatch.setattr(drift_module, "REFERENCE_PATH", tmp_path / "reference.parquet")

    feature_df = _make_feature_df(n)
    target_series = pd.Series([0] * n)

    with (
        patch("monitoring.drift.load_raw_data", return_value=pd.DataFrame()),
        patch(
            "monitoring.drift.build_features", return_value=(feature_df, target_series)
        ),
    ):
        drift_module.build_reference_dataset()

    saved = pd.read_parquet(tmp_path / "reference.parquet")
    assert len(saved) == int(n * 0.7)


def test_build_reference_dataset_contains_target_column(tmp_path, monkeypatch):
    """Saved reference must have a 'target' column."""
    import monitoring.drift as drift_module

    monkeypatch.setattr(drift_module, "REFERENCE_PATH", tmp_path / "reference.parquet")
    feature_df = _make_feature_df(50)
    target_series = pd.Series([0] * 50)

    with (
        patch("monitoring.drift.load_raw_data", return_value=pd.DataFrame()),
        patch(
            "monitoring.drift.build_features", return_value=(feature_df, target_series)
        ),
    ):
        drift_module.build_reference_dataset()

    saved = pd.read_parquet(tmp_path / "reference.parquet")
    assert "target" in saved.columns


def test_build_reference_dataset_creates_parent_dir(tmp_path, monkeypatch):
    """build_reference_dataset must create missing parent directories."""
    import monitoring.drift as drift_module

    nested = tmp_path / "a" / "b" / "reference.parquet"
    monkeypatch.setattr(drift_module, "REFERENCE_PATH", nested)
    feature_df = _make_feature_df(20)
    target_series = pd.Series([0] * 20)

    with (
        patch("monitoring.drift.load_raw_data", return_value=pd.DataFrame()),
        patch(
            "monitoring.drift.build_features", return_value=(feature_df, target_series)
        ),
    ):
        drift_module.build_reference_dataset()

    assert nested.exists()


# ---------------------------------------------------------------------------
# detect_drift tests  (mock Evidently to keep tests fast & hermetic)
# ---------------------------------------------------------------------------


def _mock_snapshot(drift_share: float) -> MagicMock:
    """Return a mock Evidently snapshot with a given drift_share."""
    snapshot = MagicMock()
    snapshot.dict.return_value = {"metrics": [{"value": {"share": drift_share}}]}
    return snapshot


def test_detect_drift_returns_dict(tmp_path, monkeypatch):
    """detect_drift must return a dict."""
    import monitoring.drift as drift_module

    ref_path = _make_reference_parquet(tmp_path)
    monkeypatch.setattr(drift_module, "REFERENCE_PATH", ref_path)
    monkeypatch.setattr(drift_module, "REPORTS_PATH", tmp_path / "reports")

    mock_report = MagicMock()
    mock_snapshot = _mock_snapshot(0.0)
    mock_report.run.return_value = mock_snapshot

    with patch("monitoring.drift.Report", return_value=mock_report):
        production = _make_feature_df(30)
        result = drift_module.detect_drift(production)

    assert isinstance(result, dict)


def test_detect_drift_has_required_keys(tmp_path, monkeypatch):
    """Result dict must contain 'drift_detected' and 'report_path'."""
    import monitoring.drift as drift_module

    ref_path = _make_reference_parquet(tmp_path)
    monkeypatch.setattr(drift_module, "REFERENCE_PATH", ref_path)
    monkeypatch.setattr(drift_module, "REPORTS_PATH", tmp_path / "reports")

    mock_report = MagicMock()
    mock_report.run.return_value = _mock_snapshot(0.0)

    with patch("monitoring.drift.Report", return_value=mock_report):
        result = drift_module.detect_drift(_make_feature_df(30))

    assert "drift_detected" in result
    assert "report_path" in result


def test_detect_drift_is_false_below_threshold(tmp_path, monkeypatch):
    """drift_detected must be False when drift_share < threshold."""
    import monitoring.drift as drift_module

    ref_path = _make_reference_parquet(tmp_path)
    monkeypatch.setattr(drift_module, "REFERENCE_PATH", ref_path)
    monkeypatch.setattr(drift_module, "REPORTS_PATH", tmp_path / "reports")
    monkeypatch.setattr(drift_module, "DRIFT_THRESHOLD", 0.05)

    mock_report = MagicMock()
    mock_report.run.return_value = _mock_snapshot(0.01)  # below threshold

    with patch("monitoring.drift.Report", return_value=mock_report):
        result = drift_module.detect_drift(_make_feature_df(30))

    assert result["drift_detected"] is False


def test_detect_drift_is_true_at_threshold(tmp_path, monkeypatch):
    """drift_detected must be True when drift_share >= threshold."""
    import monitoring.drift as drift_module

    ref_path = _make_reference_parquet(tmp_path)
    monkeypatch.setattr(drift_module, "REFERENCE_PATH", ref_path)
    monkeypatch.setattr(drift_module, "REPORTS_PATH", tmp_path / "reports")
    monkeypatch.setattr(drift_module, "DRIFT_THRESHOLD", 0.05)

    mock_report = MagicMock()
    mock_report.run.return_value = _mock_snapshot(0.05)  # exactly at threshold

    with patch("monitoring.drift.Report", return_value=mock_report):
        result = drift_module.detect_drift(_make_feature_df(30))

    assert result["drift_detected"] is True


def test_detect_drift_is_true_above_threshold(tmp_path, monkeypatch):
    """drift_detected must be True when drift_share > threshold."""
    import monitoring.drift as drift_module

    ref_path = _make_reference_parquet(tmp_path)
    monkeypatch.setattr(drift_module, "REFERENCE_PATH", ref_path)
    monkeypatch.setattr(drift_module, "REPORTS_PATH", tmp_path / "reports")
    monkeypatch.setattr(drift_module, "DRIFT_THRESHOLD", 0.05)

    mock_report = MagicMock()
    mock_report.run.return_value = _mock_snapshot(0.50)  # heavy drift

    with patch("monitoring.drift.Report", return_value=mock_report):
        result = drift_module.detect_drift(_make_feature_df(30))

    assert result["drift_detected"] is True


def test_detect_drift_report_path_is_html(tmp_path, monkeypatch):
    """Report path must end with .html."""
    import monitoring.drift as drift_module

    ref_path = _make_reference_parquet(tmp_path)
    monkeypatch.setattr(drift_module, "REFERENCE_PATH", ref_path)
    monkeypatch.setattr(drift_module, "REPORTS_PATH", tmp_path / "reports")

    mock_report = MagicMock()
    mock_report.run.return_value = _mock_snapshot(0.0)

    with patch("monitoring.drift.Report", return_value=mock_report):
        result = drift_module.detect_drift(_make_feature_df(30))

    assert result["report_path"].endswith(".html")


def test_detect_drift_calls_evidently_report(tmp_path, monkeypatch):
    """detect_drift must call Report.run() exactly once."""
    import monitoring.drift as drift_module

    ref_path = _make_reference_parquet(tmp_path)
    monkeypatch.setattr(drift_module, "REFERENCE_PATH", ref_path)
    monkeypatch.setattr(drift_module, "REPORTS_PATH", tmp_path / "reports")

    mock_report = MagicMock()
    mock_report.run.return_value = _mock_snapshot(0.0)

    with patch("monitoring.drift.Report", return_value=mock_report):
        drift_module.detect_drift(_make_feature_df(30))

    mock_report.run.assert_called_once()
