"""Tests for src/pipelines/training.py — unit tests for individual Prefect tasks.

Prefect tasks are tested by calling the underlying function directly (without
the Prefect runtime) via the .fn attribute, so these tests run fast without
needing a running Prefect server.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import yaml
from conftest import make_raw_csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_df(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Air_temperature_K": rng.normal(300, 2, rows),
            "Process_temperature_K": rng.normal(310, 2, rows),
            "Rotational_speed_rpm": rng.integers(1400, 1700, rows).astype(float),
            "Torque_Nm": rng.normal(40, 5, rows),
            "Tool_wear_min": rng.integers(0, 200, rows).astype(float),
            "Type_H": rng.integers(0, 2, rows),
            "Type_L": rng.integers(0, 2, rows),
            "Type_M": rng.integers(0, 2, rows),
        }
    )


def _make_target(rows: int = 200) -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.integers(0, 2, rows))


def _make_training_config(tmp_path: Path) -> dict:
    """Return a minimal training config dict wired to a temp raw CSV."""
    csv_path = make_raw_csv(tmp_path, rows=102)
    return {
        "data": {
            "raw_path": str(csv_path),
            "processed_path": str(tmp_path / "features.parquet"),
            "test_size": 0.2,
            "random_state": 42,
        },
        "model": {
            "params": {
                "n_estimators": 5,
                "max_depth": 2,
                "learning_rate": 0.1,
                "scale_pos_weight": 10,
            }
        },
    }


# ---------------------------------------------------------------------------
# load_config_task task
# ---------------------------------------------------------------------------


def test_load_config_task_returns_dict(tmp_path, monkeypatch):
    """load_config_task must return a dict."""
    import core.config as config_module
    from pipelines.training import load_config_task

    config_content = {"data": {"raw_path": "x.csv"}}
    cfg_path = tmp_path / "training.yaml"
    cfg_path.write_text(yaml.dump(config_content))

    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)

    result = load_config_task.fn()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# ingest_data task
# ---------------------------------------------------------------------------


def test_ingest_data_task_returns_dataframe(tmp_path):
    """ingest_data task must return a pd.DataFrame."""
    from pipelines.training import ingest_data

    config = _make_training_config(tmp_path)
    result = ingest_data.fn(config)
    assert isinstance(result, pd.DataFrame)


def test_ingest_data_task_has_expected_rows(tmp_path):
    """ingest_data task must return 102 rows (our minimal CSV)."""
    from pipelines.training import ingest_data

    config = _make_training_config(tmp_path)
    result = ingest_data.fn(config)
    assert len(result) == 102


# ---------------------------------------------------------------------------
# build_features_task
# ---------------------------------------------------------------------------


def test_build_features_task_returns_tuple(tmp_path):
    """build_features_task must return a (DataFrame, Series) tuple."""
    from pipelines.training import build_features_task, ingest_data

    config = _make_training_config(tmp_path)
    df = ingest_data.fn(config)
    X, y = build_features_task.fn(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_build_features_task_feature_count(tmp_path):
    """Feature matrix must have 8 columns."""
    from pipelines.training import build_features_task, ingest_data

    config = _make_training_config(tmp_path)
    df = ingest_data.fn(config)
    X, _ = build_features_task.fn(df)
    assert X.shape[1] == 8


# ---------------------------------------------------------------------------
# split_data task
# ---------------------------------------------------------------------------


def test_split_data_task_returns_four_parts():
    """split_data task must return four DataFrames/Series."""
    from pipelines.training import split_data

    X = _make_feature_df()
    y = _make_target()
    config = {"data": {"test_size": 0.2, "random_state": 42}}

    X_train, X_test, y_train, y_test = split_data.fn(X, y, config)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


def test_split_data_task_sizes():
    """80% train / 20% test split for 200 rows."""
    from pipelines.training import split_data

    X = _make_feature_df(200)
    y = _make_target(200)
    config = {"data": {"test_size": 0.2, "random_state": 42}}

    X_train, X_test, y_train, y_test = split_data.fn(X, y, config)
    assert len(X_train) == 160
    assert len(X_test) == 40


def test_split_data_task_no_overlap():
    """Train and test sets must not share indices."""
    from pipelines.training import split_data

    X = _make_feature_df(100)
    y = _make_target(100)
    config = {"data": {"test_size": 0.2, "random_state": 42}}

    X_train, X_test, _, _ = split_data.fn(X, y, config)
    assert len(set(X_train.index) & set(X_test.index)) == 0


def test_split_data_task_stratified():
    """Stratified split must preserve class ratio (within 10%)."""
    from pipelines.training import split_data

    n = 200
    y = pd.Series([0] * 180 + [1] * 20)
    X = _make_feature_df(n)
    config = {"data": {"test_size": 0.2, "random_state": 0}}

    _, _, y_train, y_test = split_data.fn(X, y, config)
    assert abs(y_train.mean() - y_test.mean()) < 0.10


# ---------------------------------------------------------------------------
# train_model task
# ---------------------------------------------------------------------------


def test_train_model_task_returns_classifier(tmp_path):
    """train_model task must return an XGBClassifier instance."""
    from xgboost import XGBClassifier

    from pipelines.training import split_data, train_model

    config = _make_training_config(tmp_path)
    X = _make_feature_df(200)
    y = _make_target(200)
    split_cfg = {"data": {"test_size": 0.2, "random_state": 42}}
    X_train, _, y_train, _ = split_data.fn(X, y, split_cfg)

    model = train_model.fn(X_train, y_train, config)
    assert isinstance(model, XGBClassifier)


def test_train_model_task_is_fitted(tmp_path):
    """Returned model must be fitted (can call predict)."""
    from pipelines.training import split_data, train_model

    config = _make_training_config(tmp_path)
    X = _make_feature_df(200)
    y = _make_target(200)
    split_cfg = {"data": {"test_size": 0.2, "random_state": 42}}
    X_train, X_test, y_train, _ = split_data.fn(X, y, split_cfg)

    model = train_model.fn(X_train, y_train, config)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


# ---------------------------------------------------------------------------
# evaluate_model task
# ---------------------------------------------------------------------------


def test_evaluate_model_task_returns_dict(tmp_path):
    """evaluate_model task must return a dict."""
    from pipelines.training import evaluate_model, split_data, train_model

    config = _make_training_config(tmp_path)
    X = _make_feature_df(200)
    y = _make_target(200)
    split_cfg = {"data": {"test_size": 0.2, "random_state": 42}}
    X_train, X_test, y_train, y_test = split_data.fn(X, y, split_cfg)
    model = train_model.fn(X_train, y_train, config)

    metrics = evaluate_model.fn(model, X_test, y_test)
    assert isinstance(metrics, dict)


def test_evaluate_model_task_has_required_keys(tmp_path):
    """Metrics dict must contain f1, precision, recall, roc_auc."""
    from pipelines.training import evaluate_model, split_data, train_model

    config = _make_training_config(tmp_path)
    X = _make_feature_df(200)
    y = _make_target(200)
    split_cfg = {"data": {"test_size": 0.2, "random_state": 42}}
    X_train, X_test, y_train, y_test = split_data.fn(X, y, split_cfg)
    model = train_model.fn(X_train, y_train, config)

    metrics = evaluate_model.fn(model, X_test, y_test)
    for key in ("f1", "precision", "recall", "roc_auc"):
        assert key in metrics, f"Missing metric: {key}"


def test_evaluate_model_task_values_in_range(tmp_path):
    """All metric values must be floats in [0, 1]."""
    from pipelines.training import evaluate_model, split_data, train_model

    config = _make_training_config(tmp_path)
    X = _make_feature_df(200)
    y = _make_target(200)
    split_cfg = {"data": {"test_size": 0.2, "random_state": 42}}
    X_train, X_test, y_train, y_test = split_data.fn(X, y, split_cfg)
    model = train_model.fn(X_train, y_train, config)

    metrics = evaluate_model.fn(model, X_test, y_test)
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"Metric {k}={v} out of [0,1]"


# ---------------------------------------------------------------------------
# log_to_mlflow task
# ---------------------------------------------------------------------------


def test_log_to_mlflow_task_calls_log_metrics():
    """log_to_mlflow task must call mlflow.log_metrics with all 4 metrics."""
    from pipelines.training import log_to_mlflow

    metrics = {"f1": 0.8, "precision": 0.75, "recall": 0.85, "roc_auc": 0.9}

    with (
        patch("pipelines.training.mlflow.set_tracking_uri"),
        patch("pipelines.training.mlflow.set_experiment"),
        patch("pipelines.training.mlflow.start_run") as mock_run,
        patch("pipelines.training.mlflow.log_metrics") as mock_log,
        patch("pipelines.training.mlflow.xgboost.log_model"),
    ):
        mock_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_run.return_value.__exit__ = MagicMock(return_value=False)

        log_to_mlflow.fn(MagicMock(), metrics)

    mock_log.assert_called_once_with(metrics)


def test_log_to_mlflow_task_registers_model():
    """log_to_mlflow task must call mlflow.xgboost.log_model."""
    from pipelines.training import log_to_mlflow

    metrics = {"f1": 0.8, "precision": 0.75, "recall": 0.85, "roc_auc": 0.9}
    mock_model = MagicMock()

    with (
        patch("pipelines.training.mlflow.set_tracking_uri"),
        patch("pipelines.training.mlflow.set_experiment"),
        patch("pipelines.training.mlflow.start_run") as mock_run,
        patch("pipelines.training.mlflow.log_metrics"),
        patch("pipelines.training.mlflow.xgboost.log_model") as mock_xgb_log,
    ):
        mock_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_run.return_value.__exit__ = MagicMock(return_value=False)

        log_to_mlflow.fn(mock_model, metrics)

    mock_xgb_log.assert_called_once()
    assert mock_xgb_log.call_args[0][0] is mock_model


# ---------------------------------------------------------------------------
# save_model task
# ---------------------------------------------------------------------------


def test_save_model_task_saves_pkl(tmp_path, monkeypatch):
    """save_model task must write a .pkl file via joblib.dump."""
    import pipelines.training as train_module

    monkeypatch.setattr(train_module, "MODEL_PATH", tmp_path / "model.pkl")
    mock_model = MagicMock()

    with patch("pipelines.training.joblib.dump") as mock_dump:
        train_module.save_model.fn(mock_model)

    mock_dump.assert_called_once()
    assert mock_dump.call_args[0][0] is mock_model


def test_save_model_task_uses_model_path(tmp_path, monkeypatch):
    """save_model task must dump to MODEL_PATH."""
    import pipelines.training as train_module

    expected_path = tmp_path / "model.pkl"
    monkeypatch.setattr(train_module, "MODEL_PATH", expected_path)
    mock_model = MagicMock()

    with patch("pipelines.training.joblib.dump") as mock_dump:
        train_module.save_model.fn(mock_model)

    assert mock_dump.call_args[0][1] == expected_path
