"""Tests for src/pipelines/data.py."""

from pathlib import Path

import pandas as pd
import pytest
import yaml
from conftest import make_raw_csv

from core.config import load_config
from pipelines.data import run_pipeline

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


# ---------------------------------------------------------------------------
# load_config tests
# ---------------------------------------------------------------------------


def test_load_config_returns_valid_structure(tmp_path):
    """load_config must return a dict with a 'data' section containing required keys."""
    cfg_path = _write_config(tmp_path, "my/raw.csv", "my/out.parquet")
    config = load_config(cfg_path)

    assert isinstance(config, dict)
    assert "data" in config
    for key in ("raw_path", "processed_path", "test_size", "random_state"):
        assert key in config["data"], f"Key missing from config: {key}"


def test_load_config_values_are_correct(tmp_path):
    """Values must match what was written to the YAML."""
    cfg_path = _write_config(tmp_path, "my/raw.csv", "my/out.parquet")
    config = load_config(cfg_path)

    assert config["data"]["raw_path"] == "my/raw.csv"
    assert config["data"]["processed_path"] == "my/out.parquet"
    assert config["data"]["test_size"] == pytest.approx(0.2)
    assert config["data"]["random_state"] == 42


def test_load_config_raises_for_missing_file():
    """FileNotFoundError when config path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("nonexistent/config.yaml"))


# ---------------------------------------------------------------------------
# run_pipeline fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_config(tmp_path):
    """Config dict wired to a small raw CSV and a temp output path."""
    raw_path = make_raw_csv(tmp_path)
    out_path = tmp_path / "processed" / "features.parquet"
    return {
        "data": {
            "raw_path": str(raw_path),
            "processed_path": str(out_path),
        }
    }


# ---------------------------------------------------------------------------
# run_pipeline tests
# ---------------------------------------------------------------------------


def test_run_pipeline_returns_tuple(pipeline_config):
    """run_pipeline must return (DataFrame, Series)."""
    X, y = run_pipeline(pipeline_config)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_run_pipeline_creates_parquet(pipeline_config):
    """run_pipeline must write a Parquet file at the configured path."""
    run_pipeline(pipeline_config)
    out_path = Path(pipeline_config["data"]["processed_path"])
    assert out_path.exists(), "Parquet file was not created"


def test_run_pipeline_parquet_readable(tmp_path):
    """Output Parquet file must be readable and have the same row count as input."""
    rows = 30
    raw_path = make_raw_csv(tmp_path, rows=rows)
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
    assert len(df) == rows


def test_run_pipeline_parquet_has_target_column(pipeline_config):
    """Saved Parquet must include a 'target' column."""
    run_pipeline(pipeline_config)
    df = pd.read_parquet(pipeline_config["data"]["processed_path"])
    assert "target" in df.columns


def test_run_pipeline_creates_parent_directories(tmp_path):
    """run_pipeline must create missing intermediate directories."""
    raw_path = make_raw_csv(tmp_path)
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
    raw_path = make_raw_csv(tmp_path, rows=rows)
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
