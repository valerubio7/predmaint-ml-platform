"""Tests for src/pipelines/monitoring.py — check_drift task and monitoring_pipeline."""

from unittest.mock import patch

import pandas as pd
from conftest import make_raw_csv

# ---------------------------------------------------------------------------
# _read_bool_env tests
# ---------------------------------------------------------------------------


def test_read_bool_env_returns_default_when_unset(monkeypatch):
    """_read_bool_env must return the default when the variable is not set."""
    import pipelines.monitoring as monitor_module

    monkeypatch.delenv("TEST_BOOL_VAR", raising=False)
    assert monitor_module._read_bool_env("TEST_BOOL_VAR", default=True) is True
    assert monitor_module._read_bool_env("TEST_BOOL_VAR", default=False) is False


def test_read_bool_env_parses_true(monkeypatch):
    """_read_bool_env must parse 'true' (case-insensitive) as True."""
    import pipelines.monitoring as monitor_module

    for value in ("true", "True", "TRUE", "  true  "):
        monkeypatch.setenv("TEST_BOOL_VAR", value)
        assert monitor_module._read_bool_env("TEST_BOOL_VAR", default=False) is True


def test_read_bool_env_parses_false(monkeypatch):
    """_read_bool_env must parse 'false' (case-insensitive) as False."""
    import pipelines.monitoring as monitor_module

    for value in ("false", "False", "FALSE", "  false  "):
        monkeypatch.setenv("TEST_BOOL_VAR", value)
        assert monitor_module._read_bool_env("TEST_BOOL_VAR", default=True) is False


def test_read_bool_env_raises_on_invalid_value(monkeypatch):
    """_read_bool_env must raise ValueError for non-boolean strings."""
    import pipelines.monitoring as monitor_module

    monkeypatch.setenv("TEST_BOOL_VAR", "yes")
    try:
        monitor_module._read_bool_env("TEST_BOOL_VAR", default=False)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "TEST_BOOL_VAR" in str(exc)
        assert "yes" in str(exc)


# ---------------------------------------------------------------------------
# _build_production_sample tests
# ---------------------------------------------------------------------------


def test_build_production_sample_returns_dataframe(tmp_path):
    """_build_production_sample must return a DataFrame."""
    import pipelines.monitoring as monitor_module

    rows_total = 100
    feature_df = pd.DataFrame({"a": range(rows_total)})

    with patch(
        "pipelines.monitoring.load_features",
        return_value=(feature_df, pd.Series([0] * rows_total)),
    ):
        result = monitor_module._build_production_sample(tmp_path / "raw.csv")

    assert isinstance(result, pd.DataFrame)


def test_build_production_sample_slices_after_reference_split(tmp_path):
    """Production sample must contain only rows after the reference split point."""
    import pipelines.monitoring as monitor_module
    from monitoring.drift.detector import REFERENCE_SPLIT_RATIO, _compute_split_index

    rows_total = 100
    feature_df = pd.DataFrame({"a": range(rows_total)})
    expected_split = _compute_split_index(rows_total, REFERENCE_SPLIT_RATIO)

    with patch(
        "pipelines.monitoring.load_features",
        return_value=(feature_df, pd.Series([0] * rows_total)),
    ):
        result = monitor_module._build_production_sample(tmp_path / "raw.csv")

    assert len(result) == rows_total - expected_split
    assert list(result["a"]) == list(range(expected_split, rows_total))


# ---------------------------------------------------------------------------
# check_drift task tests
# ---------------------------------------------------------------------------


def test_check_drift_returns_bool(tmp_path):
    """check_drift task must return a bool."""
    import pipelines.monitoring as monitor_module

    config = {"data": {"raw_path": str(make_raw_csv(tmp_path))}}
    drift_result = {"drift_detected": False, "report_path": "report.html"}

    with (
        patch("pipelines.monitoring.detect_drift", return_value=drift_result),
        patch(
            "pipelines.monitoring.load_features",
            return_value=(pd.DataFrame({"a": range(100)}), pd.Series([0] * 100)),
        ),
    ):
        result = monitor_module.check_drift.fn(config)

    assert isinstance(result, bool)


def test_check_drift_returns_false_when_no_drift(tmp_path):
    """check_drift must return False when detect_drift reports no drift."""
    import pipelines.monitoring as monitor_module

    config = {"data": {"raw_path": str(make_raw_csv(tmp_path))}}
    drift_result = {"drift_detected": False, "report_path": "report.html"}

    with (
        patch("pipelines.monitoring.detect_drift", return_value=drift_result),
        patch(
            "pipelines.monitoring.load_features",
            return_value=(pd.DataFrame({"a": range(100)}), pd.Series([0] * 100)),
        ),
    ):
        result = monitor_module.check_drift.fn(config)

    assert result is False


def test_check_drift_returns_true_when_drift(tmp_path):
    """check_drift must return True when detect_drift reports drift."""
    import pipelines.monitoring as monitor_module

    config = {"data": {"raw_path": str(make_raw_csv(tmp_path))}}
    drift_result = {"drift_detected": True, "report_path": "report.html"}

    with (
        patch("pipelines.monitoring.detect_drift", return_value=drift_result),
        patch(
            "pipelines.monitoring.load_features",
            return_value=(pd.DataFrame({"a": range(100)}), pd.Series([0] * 100)),
        ),
    ):
        result = monitor_module.check_drift.fn(config)

    assert result is True


def test_check_drift_passes_production_sample_to_detect_drift(tmp_path):
    """check_drift must pass the post-reference slice to detect_drift."""
    import pipelines.monitoring as monitor_module
    from monitoring.drift.detector import REFERENCE_SPLIT_RATIO, _compute_split_index

    config = {"data": {"raw_path": str(make_raw_csv(tmp_path))}}
    rows_total = 100
    feature_df = pd.DataFrame({"a": range(rows_total)})
    expected_split = _compute_split_index(rows_total, REFERENCE_SPLIT_RATIO)
    drift_result = {"drift_detected": False, "report_path": "r.html"}

    with (
        patch(
            "pipelines.monitoring.load_features",
            return_value=(feature_df, pd.Series([0] * rows_total)),
        ),
        patch(
            "pipelines.monitoring.detect_drift", return_value=drift_result
        ) as mock_detect,
    ):
        monitor_module.check_drift.fn(config)

    passed_df: pd.DataFrame = mock_detect.call_args[0][0]
    assert len(passed_df) == rows_total - expected_split


# ---------------------------------------------------------------------------
# monitoring_pipeline flow branching tests
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
