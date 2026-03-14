"""Tests for src/core/loader.py."""

import re
from pathlib import Path

import pandas as pd
import pytest
from conftest import make_valid_csv

from core.loader import EXPECTED_COLUMNS, load_raw_data

# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_load_raw_data_returns_dataframe_with_expected_shape():
    """Default path must load the full AI4I dataset (10 000 rows, 14 columns)."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10_000, len(EXPECTED_COLUMNS))


def test_load_raw_data_has_expected_columns():
    """All 14 expected columns must be present."""
    df = load_raw_data()
    assert set(EXPECTED_COLUMNS).issubset(set(df.columns))


def test_load_raw_data_custom_path(tmp_path: Path):
    """Accepts a custom path and reads the CSV correctly."""
    csv_path = make_valid_csv(tmp_path, rows=10)
    df = load_raw_data(csv_path)
    assert df.shape == (10, len(EXPECTED_COLUMNS))


def test_load_raw_data_allows_extra_columns(tmp_path: Path):
    """Columns beyond the expected set must not cause a failure."""
    csv_path = make_valid_csv(tmp_path)
    df = pd.read_csv(csv_path)
    df["extra_col"] = 99
    df.to_csv(csv_path, index=False)

    result = load_raw_data(csv_path)

    assert "extra_col" in result.columns
    assert set(EXPECTED_COLUMNS).issubset(set(result.columns))


def test_load_raw_data_accepts_header_only_csv(tmp_path: Path):
    """A CSV with only a header and no data rows must return an empty DataFrame."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(",".join(EXPECTED_COLUMNS) + "\n")

    df = load_raw_data(csv_path)

    assert df.shape == (0, len(EXPECTED_COLUMNS))


# ---------------------------------------------------------------------------
# Error tests
# ---------------------------------------------------------------------------


def test_load_raw_data_raises_for_missing_file():
    """FileNotFoundError when the CSV path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_raw_data(Path("nonexistent/path/data.csv"))


def test_load_raw_data_raises_for_missing_columns(tmp_path: Path):
    """ValueError must be raised when required columns are absent."""
    missing_col = "Machine failure"
    data = {col: [1, 2] for col in EXPECTED_COLUMNS if col != missing_col}
    csv_path = tmp_path / "incomplete.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing columns"):
        load_raw_data(csv_path)


def test_load_raw_data_error_message_names_missing_column(tmp_path: Path):
    """The ValueError message must include the name of the missing column."""
    missing_col = "Torque [Nm]"
    data = {col: [1] for col in EXPECTED_COLUMNS if col != missing_col}
    csv_path = tmp_path / "partial.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match=re.escape(missing_col)):
        load_raw_data(csv_path)


def test_load_raw_data_raises_for_completely_unrelated_columns(tmp_path: Path):
    """ValueError when the CSV has no columns in common with the expected schema."""
    csv_path = tmp_path / "wrong.csv"
    pd.DataFrame({"col_a": [1], "col_b": [2]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing columns"):
        load_raw_data(csv_path)
