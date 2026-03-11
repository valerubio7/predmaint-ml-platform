"""Tests for src/training/train.py — unit tests for individual Prefect tasks.

Prefect tasks are tested by calling the underlying function directly (without
the Prefect runtime) via the .fn attribute, so these tests run fast without
needing a running Prefect server.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import yaml

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


def _make_config(tmp_path: Path) -> dict:
    """Return a minimal config dict pointing to a temp raw CSV."""

    types = ["L", "M", "H"] * 34  # 102 rows
    data = {
        "UDI": list(range(1, 103)),
        "Product ID": [f"M{i}" for i in range(102)],
        "Type": types[:102],
        "Air temperature [K]": [300.0] * 102,
        "Process temperature [K]": [310.0] * 102,
        "Rotational speed [rpm]": [1500] * 102,
        "Torque [Nm]": [40.0] * 102,
        "Tool wear [min]": [0] * 102,
        "Machine failure": [0] * 95 + [1] * 7,
        "TWF": [0] * 102,
        "HDF": [0] * 102,
        "PWF": [0] * 102,
        "OSF": [0] * 102,
        "RNF": [0] * 102,
    }
    csv_path = tmp_path / "raw.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

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
# load_config task
# ---------------------------------------------------------------------------


def test_load_config_task_returns_dict(tmp_path, monkeypatch):
    """load_config task must return a dict."""
    from training.train import load_config

    config_content = {"data": {"raw_path": "x.csv"}}
    cfg_path = tmp_path / "training.yaml"
    cfg_path.write_text(yaml.dump(config_content))

    import training.train as train_module

    monkeypatch.setattr(train_module, "CONFIG_PATH", cfg_path)

    result = load_config.fn()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# ingest_data task
# ---------------------------------------------------------------------------


def test_ingest_data_task_returns_dataframe(tmp_path):
    """ingest_data task must return a pd.DataFrame."""
    from training.train import ingest_data

    config = _make_config(tmp_path)
    result = ingest_data.fn(config)
    assert isinstance(result, pd.DataFrame)


def test_ingest_data_task_has_expected_rows(tmp_path):
    """ingest_data task must return 102 rows (our minimal CSV)."""
    from training.train import ingest_data

    config = _make_config(tmp_path)
    result = ingest_data.fn(config)
    assert len(result) == 102


# ---------------------------------------------------------------------------
# build_features_task
# ---------------------------------------------------------------------------


def test_build_features_task_returns_tuple(tmp_path):
    """build_features_task must return a (DataFrame, Series) tuple."""
    from training.train import build_features_task, ingest_data

    config = _make_config(tmp_path)
    df = ingest_data.fn(config)
    X, y = build_features_task.fn(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_build_features_task_feature_count(tmp_path):
    """Feature matrix must have 8 columns."""
    from training.train import build_features_task, ingest_data

    config = _make_config(tmp_path)
    df = ingest_data.fn(config)
    X, _ = build_features_task.fn(df)
    assert X.shape[1] == 8


# ---------------------------------------------------------------------------
# split_data task
# ---------------------------------------------------------------------------


def test_split_data_task_returns_four_parts():
    """split_data task must return four DataFrames/Series."""
    from training.train import split_data

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
    from training.train import split_data

    X = _make_feature_df(200)
    y = _make_target(200)
    config = {"data": {"test_size": 0.2, "random_state": 42}}

    X_train, X_test, y_train, y_test = split_data.fn(X, y, config)
    assert len(X_train) == 160
    assert len(X_test) == 40


def test_split_data_task_no_overlap():
    """Train and test sets must not share indices."""
    from training.train import split_data

    X = _make_feature_df(100)
    y = _make_target(100)
    config = {"data": {"test_size": 0.2, "random_state": 42}}

    X_train, X_test, _, _ = split_data.fn(X, y, config)
    assert len(set(X_train.index) & set(X_test.index)) == 0


def test_split_data_task_stratified():
    """Stratified split must preserve class ratio (within 10%)."""
    from training.train import split_data

    n = 200
    # Imbalanced: 90% 0, 10% 1
    y = pd.Series([0] * 180 + [1] * 20)
    X = _make_feature_df(n)
    config = {"data": {"test_size": 0.2, "random_state": 0}}

    _, _, y_train, y_test = split_data.fn(X, y, config)
    train_pos_rate = y_train.mean()
    test_pos_rate = y_test.mean()
    assert abs(train_pos_rate - test_pos_rate) < 0.10


# ---------------------------------------------------------------------------
# train_model task
# ---------------------------------------------------------------------------


def test_train_model_task_returns_classifier(tmp_path):
    """train_model task must return an XGBClassifier instance."""
    from xgboost import XGBClassifier

    from training.train import split_data, train_model

    config = _make_config(tmp_path)
    X = _make_feature_df(200)
    y = _make_target(200)
    split_cfg = {"data": {"test_size": 0.2, "random_state": 42}}
    X_train, _, y_train, _ = split_data.fn(X, y, split_cfg)

    model = train_model.fn(X_train, y_train, config)
    assert isinstance(model, XGBClassifier)


def test_train_model_task_is_fitted(tmp_path):
    """Returned model must be fitted (can call predict)."""
    from training.train import split_data, train_model

    config = _make_config(tmp_path)
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
    from training.train import evaluate_model, split_data, train_model

    config = _make_config(tmp_path)
    X = _make_feature_df(200)
    y = _make_target(200)
    split_cfg = {"data": {"test_size": 0.2, "random_state": 42}}
    X_train, X_test, y_train, y_test = split_data.fn(X, y, split_cfg)
    model = train_model.fn(X_train, y_train, config)

    metrics = evaluate_model.fn(model, X_test, y_test)
    assert isinstance(metrics, dict)


def test_evaluate_model_task_has_required_keys(tmp_path):
    """Metrics dict must contain f1, precision, recall, roc_auc."""
    from training.train import evaluate_model, split_data, train_model

    config = _make_config(tmp_path)
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
    from training.train import evaluate_model, split_data, train_model

    config = _make_config(tmp_path)
    X = _make_feature_df(200)
    y = _make_target(200)
    split_cfg = {"data": {"test_size": 0.2, "random_state": 42}}
    X_train, X_test, y_train, y_test = split_data.fn(X, y, split_cfg)
    model = train_model.fn(X_train, y_train, config)

    metrics = evaluate_model.fn(model, X_test, y_test)
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"Metric {k}={v} out of [0,1]"


# ---------------------------------------------------------------------------
# save_model task
# ---------------------------------------------------------------------------


def test_save_model_task_saves_pkl(tmp_path, monkeypatch):
    """save_model task must write a .pkl file to MODEL_PATH."""
    import training.train as train_module

    monkeypatch.setattr(train_module, "MODEL_PATH", tmp_path / "model.pkl")

    mock_model = MagicMock()
    metrics = {"f1": 0.8, "precision": 0.75, "recall": 0.85, "roc_auc": 0.9}

    with (
        patch("training.train.mlflow.set_tracking_uri"),
        patch("training.train.mlflow.set_experiment"),
        patch("training.train.mlflow.start_run") as mock_run,
        patch("training.train.mlflow.log_metrics"),
        patch("training.train.mlflow.xgboost.log_model"),
        patch("training.train.joblib.dump") as mock_dump,
    ):
        mock_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_run.return_value.__exit__ = MagicMock(return_value=False)

        train_module.save_model.fn(mock_model, metrics)

    mock_dump.assert_called_once()
    # First argument to joblib.dump must be the model
    assert mock_dump.call_args[0][0] is mock_model


def test_save_model_task_logs_all_metrics(tmp_path, monkeypatch):
    """save_model task must call mlflow.log_metrics with all 4 metrics."""
    import training.train as train_module

    monkeypatch.setattr(train_module, "MODEL_PATH", tmp_path / "model.pkl")

    metrics = {"f1": 0.8, "precision": 0.75, "recall": 0.85, "roc_auc": 0.9}

    with (
        patch("training.train.mlflow.set_tracking_uri"),
        patch("training.train.mlflow.set_experiment"),
        patch("training.train.mlflow.start_run") as mock_run,
        patch("training.train.mlflow.log_metrics") as mock_log,
        patch("training.train.mlflow.xgboost.log_model"),
        patch("training.train.joblib.dump"),
    ):
        mock_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_run.return_value.__exit__ = MagicMock(return_value=False)

        train_module.save_model.fn(MagicMock(), metrics)

    mock_log.assert_called_once_with(metrics)
