"""Tests for src/pipelines/monitoring.py — check_drift task and monitoring_pipeline."""

from unittest.mock import patch

import pandas as pd
from conftest import make_raw_csv

# ---------------------------------------------------------------------------
# check_drift task tests
# ---------------------------------------------------------------------------


def test_check_drift_returns_bool(tmp_path, monkeypatch):
    """check_drift task must return a bool."""
    import pipelines.monitoring as monitor_module

    config = {"data": {"raw_path": str(make_raw_csv(tmp_path))}}
    drift_result = {"drift_detected": False, "report_path": "report.html"}

    with (
        patch("pipelines.monitoring.detect_drift", return_value=drift_result),
        patch("pipelines.monitoring.load_features") as mock_lf,
    ):
        mock_lf.return_value = (pd.DataFrame({"a": range(8000)}), pd.Series([0] * 8000))
        result = monitor_module.check_drift.fn(config)

    assert isinstance(result, bool)


def test_check_drift_returns_false_when_no_drift(tmp_path):
    """check_drift must return False when detect_drift reports no drift."""
    import pipelines.monitoring as monitor_module

    config = {"data": {"raw_path": str(make_raw_csv(tmp_path))}}
    drift_result = {"drift_detected": False, "report_path": "report.html"}

    with (
        patch("pipelines.monitoring.detect_drift", return_value=drift_result),
        patch("pipelines.monitoring.load_features") as mock_lf,
    ):
        mock_lf.return_value = (pd.DataFrame({"a": range(8000)}), pd.Series([0] * 8000))
        result = monitor_module.check_drift.fn(config)

    assert result is False


def test_check_drift_returns_true_when_drift(tmp_path):
    """check_drift must return True when detect_drift reports drift."""
    import pipelines.monitoring as monitor_module

    config = {"data": {"raw_path": str(make_raw_csv(tmp_path))}}
    drift_result = {"drift_detected": True, "report_path": "report.html"}

    with (
        patch("pipelines.monitoring.detect_drift", return_value=drift_result),
        patch("pipelines.monitoring.load_features") as mock_lf,
    ):
        mock_lf.return_value = (pd.DataFrame({"a": range(8000)}), pd.Series([0] * 8000))
        result = monitor_module.check_drift.fn(config)

    assert result is True


def test_check_drift_slices_production_sample(tmp_path, monkeypatch):
    """check_drift must pass rows after PRODUCTION_SAMPLE_START_ROW to detect_drift."""
    import pipelines.monitoring as monitor_module

    monkeypatch.setattr(monitor_module, "PRODUCTION_SAMPLE_START_ROW", 5)
    config = {"data": {"raw_path": str(make_raw_csv(tmp_path))}}

    rows_total = 10
    feature_df = pd.DataFrame({"a": range(rows_total)})
    drift_result = {"drift_detected": False, "report_path": "r.html"}

    with (
        patch(
            "pipelines.monitoring.load_features", return_value=(feature_df, pd.Series())
        ),
        patch(
            "pipelines.monitoring.detect_drift", return_value=drift_result
        ) as mock_detect,
    ):
        monitor_module.check_drift.fn(config)

    passed_df: pd.DataFrame = mock_detect.call_args[0][0]
    assert len(passed_df) == rows_total - 5


# ---------------------------------------------------------------------------
# monitoring_pipeline flow branching tests
# (test the logic by mocking the Prefect tasks as plain callables)
# ---------------------------------------------------------------------------


def test_monitoring_pipeline_triggers_retraining_on_drift(monkeypatch):
    """When drift=True and RETRAINING_TRIGGER=True, training_pipeline must be called."""
    import pipelines.monitoring as monitor_module

    monkeypatch.setattr(monitor_module, "RETRAINING_TRIGGER", True)

    with (
        patch("pipelines.monitoring.load_config_task", return_value={}),
        patch("pipelines.monitoring.check_drift", return_value=True),
        patch("pipelines.monitoring.training_pipeline") as mock_train,
    ):
        monitor_module.monitoring_pipeline()

    mock_train.assert_called_once()


def test_monitoring_pipeline_skips_retraining_when_trigger_disabled(monkeypatch):
    """When drift=True but RETRAINING_TRIGGER=False, training_pipeline must not run."""
    import pipelines.monitoring as monitor_module

    monkeypatch.setattr(monitor_module, "RETRAINING_TRIGGER", False)

    with (
        patch("pipelines.monitoring.load_config_task", return_value={}),
        patch("pipelines.monitoring.check_drift", return_value=True),
        patch("pipelines.monitoring.training_pipeline") as mock_train,
    ):
        monitor_module.monitoring_pipeline()

    mock_train.assert_not_called()


def test_monitoring_pipeline_skips_retraining_when_no_drift(monkeypatch):
    """When drift=False, training_pipeline must not run regardless of trigger."""
    import pipelines.monitoring as monitor_module

    monkeypatch.setattr(monitor_module, "RETRAINING_TRIGGER", True)

    with (
        patch("pipelines.monitoring.load_config_task", return_value={}),
        patch("pipelines.monitoring.check_drift", return_value=False),
        patch("pipelines.monitoring.training_pipeline") as mock_train,
    ):
        monitor_module.monitoring_pipeline()

    mock_train.assert_not_called()
