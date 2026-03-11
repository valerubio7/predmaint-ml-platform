"""Tests for src/data/pipeline.py."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from data.pipeline import load_config, run_pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, raw_path: str, processed_path: str) -> Path:
    config = {
        "data": {
            "raw_path": raw_path,
            "processed_path": processed_path,
            "test_size": 0.2,
            "random_state": 42,
        }
    }
    cfg_path = tmp_path / "training.yaml"
    cfg_path.write_text(yaml.dump(config))
    return cfg_path


def _make_raw_csv(tmp_path: Path, rows: int = 20) -> Path:
    """Minimal valid CSV that satisfies loader + transformer."""

    types = ["L"] * (rows // 3) + ["M"] * (rows // 3) + ["H"] * (rows - 2 * (rows // 3))
    data: dict = {
        "UDI": list(range(1, rows + 1)),
        "Product ID": [f"M{i}" for i in range(rows)],
        "Type": types[:rows],
        "Air temperature [K]": [300.0] * rows,
        "Process temperature [K]": [310.0] * rows,
        "Rotational speed [rpm]": [1500] * rows,
        "Torque [Nm]": [40.0] * rows,
        "Tool wear [min]": [0] * rows,
        "Machine failure": [0] * (rows - 1) + [1],
        "TWF": [0] * rows,
        "HDF": [0] * rows,
        "PWF": [0] * rows,
        "OSF": [0] * rows,
        "RNF": [0] * rows,
    }
    csv_path = tmp_path / "raw.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# load_config tests
# ---------------------------------------------------------------------------


def test_load_config_returns_dict(tmp_path):
    """load_config must return a dictionary."""
    cfg_path = _write_config(
        tmp_path, "data/raw/data.csv", "data/processed/out.parquet"
    )
    config = load_config(cfg_path)
    assert isinstance(config, dict)


def test_load_config_has_data_key(tmp_path):
    """Loaded config must contain 'data' key."""
    cfg_path = _write_config(
        tmp_path, "data/raw/data.csv", "data/processed/out.parquet"
    )
    config = load_config(cfg_path)
    assert "data" in config


def test_load_config_data_section_has_required_keys(tmp_path):
    """data section must contain raw_path, processed_path, test_size, random_state."""
    cfg_path = _write_config(
        tmp_path, "data/raw/data.csv", "data/processed/out.parquet"
    )
    config = load_config(cfg_path)
    for key in ("raw_path", "processed_path", "test_size", "random_state"):
        assert key in config["data"], f"Key missing from config: {key}"


def test_load_config_raises_for_missing_file():
    """FileNotFoundError when config path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("nonexistent/config.yaml"))


def test_load_config_values_are_correct(tmp_path):
    """Values must match what was written to the YAML."""
    cfg_path = _write_config(tmp_path, "my/raw.csv", "my/out.parquet")
    config = load_config(cfg_path)
    assert config["data"]["raw_path"] == "my/raw.csv"
    assert config["data"]["processed_path"] == "my/out.parquet"
    assert config["data"]["test_size"] == pytest.approx(0.2)
    assert config["data"]["random_state"] == 42


# ---------------------------------------------------------------------------
# run_pipeline tests
# ---------------------------------------------------------------------------


def test_run_pipeline_returns_tuple(tmp_path):
    """run_pipeline must return (DataFrame, Series)."""
    raw_path = _make_raw_csv(tmp_path)
    out_path = tmp_path / "processed" / "features.parquet"
    config = {
        "data": {
            "raw_path": str(raw_path),
            "processed_path": str(out_path),
        }
    }
    X, y = run_pipeline(config)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_run_pipeline_creates_parquet(tmp_path):
    """run_pipeline must write a Parquet file at the configured path."""
    raw_path = _make_raw_csv(tmp_path)
    out_path = tmp_path / "processed" / "features.parquet"
    config = {
        "data": {
            "raw_path": str(raw_path),
            "processed_path": str(out_path),
        }
    }
    run_pipeline(config)
    assert out_path.exists(), "Parquet file was not created"


def test_run_pipeline_parquet_readable(tmp_path):
    """The output Parquet file must be readable as a DataFrame."""
    raw_path = _make_raw_csv(tmp_path, rows=30)
    out_path = tmp_path / "processed" / "features.parquet"
    config = {
        "data": {
            "raw_path": str(raw_path),
            "processed_path": str(out_path),
        }
    }
    run_pipeline(config)
    df = pd.read_parquet(out_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 30


def test_run_pipeline_parquet_has_target_column(tmp_path):
    """Saved Parquet must include a 'target' column."""
    raw_path = _make_raw_csv(tmp_path)
    out_path = tmp_path / "out.parquet"
    config = {
        "data": {
            "raw_path": str(raw_path),
            "processed_path": str(out_path),
        }
    }
    run_pipeline(config)
    df = pd.read_parquet(out_path)
    assert "target" in df.columns


def test_run_pipeline_creates_parent_directories(tmp_path):
    """run_pipeline must create missing intermediate directories."""
    raw_path = _make_raw_csv(tmp_path)
    deep_path = tmp_path / "a" / "b" / "c" / "features.parquet"
    config = {
        "data": {
            "raw_path": str(raw_path),
            "processed_path": str(deep_path),
        }
    }
    run_pipeline(config)
    assert deep_path.exists()


def test_run_pipeline_feature_row_count_matches_input(tmp_path):
    """Row count of features must equal number of rows in raw CSV."""
    rows = 40
    raw_path = _make_raw_csv(tmp_path, rows=rows)
    out_path = tmp_path / "out.parquet"
    config = {
        "data": {
            "raw_path": str(raw_path),
            "processed_path": str(out_path),
        }
    }
    X, y = run_pipeline(config)
    assert len(X) == rows
    assert len(y) == rows


def test_run_pipeline_raises_for_missing_raw_file(tmp_path):
    """run_pipeline must propagate errors when raw CSV is missing."""
    out_path = tmp_path / "out.parquet"
    config = {
        "data": {
            "raw_path": str(tmp_path / "nonexistent.csv"),
            "processed_path": str(out_path),
        }
    }
    with pytest.raises(FileNotFoundError):
        run_pipeline(config)
