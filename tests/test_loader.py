"""Tests for src/data/loader.py."""

from pathlib import Path

import pandas as pd
import pytest

from data.loader import EXPECTED_COLUMNS, load_raw_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_csv(tmp_path: Path, rows: int = 5) -> Path:
    """Create a minimal valid CSV with all expected columns."""
    data = {col: [0] * rows for col in EXPECTED_COLUMNS}
    # Give Type a string value so it matches the real dataset
    data["Type"] = ["M"] * rows
    data["Product ID"] = [f"M{i}" for i in range(rows)]
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_load_raw_data_returns_dataframe():
    """load_raw_data must return a pd.DataFrame."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)


def test_load_raw_data_has_expected_columns():
    """All 14 expected columns must be present."""
    df = load_raw_data()
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"Expected column missing: {col}"


def test_load_raw_data_default_path():
    """Default path resolves to the real dataset (10 000 rows)."""
    df = load_raw_data()
    assert len(df) == 10_000


def test_load_raw_data_custom_path(tmp_path):
    """Accepts a custom path and reads the CSV correctly."""
    csv_path = _make_valid_csv(tmp_path, rows=10)
    df = load_raw_data(csv_path)
    assert df.shape == (10, len(EXPECTED_COLUMNS))


def test_load_raw_data_preserves_column_order(tmp_path):
    """Column names must include every expected column (order-agnostic)."""
    csv_path = _make_valid_csv(tmp_path)
    df = load_raw_data(csv_path)
    assert set(EXPECTED_COLUMNS).issubset(set(df.columns))


# ---------------------------------------------------------------------------
# Error / edge-case tests
# ---------------------------------------------------------------------------


def test_load_raw_data_raises_for_missing_file():
    """FileNotFoundError when the CSV path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_raw_data(Path("nonexistent/path/data.csv"))


def test_load_raw_data_raises_for_missing_columns(tmp_path):
    """ValueError must be raised when required columns are absent."""
    # Build a CSV missing the 'Machine failure' column
    incomplete_columns = [c for c in EXPECTED_COLUMNS if c != "Machine failure"]
    data = {col: [1, 2] for col in incomplete_columns}
    csv_path = tmp_path / "incomplete.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing columns"):
        load_raw_data(csv_path)


def test_load_raw_data_raises_for_all_missing_columns(tmp_path):
    """ValueError when CSV has completely unrelated columns."""
    csv_path = tmp_path / "wrong.csv"
    pd.DataFrame({"col_a": [1], "col_b": [2]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing columns"):
        load_raw_data(csv_path)


def test_load_raw_data_raises_for_empty_file(tmp_path):
    """Raises an exception when CSV has a header but no data rows."""
    # pandas returns an empty DataFrame when all columns are present — no ValueError.
    # A CSV with wrong/missing headers would raise instead.
    csv_path = tmp_path / "empty.csv"
    # Write only the correct header, no rows
    csv_path.write_text(",".join(EXPECTED_COLUMNS) + "\n")
    df = load_raw_data(csv_path)
    assert df.shape == (0, len(EXPECTED_COLUMNS))


def test_load_raw_data_extra_columns_are_allowed(tmp_path):
    """Extra columns in the CSV should not cause a failure."""
    csv_path = _make_valid_csv(tmp_path)
    # Append an extra column by rewriting
    df = pd.read_csv(csv_path)
    df["extra_col"] = 99
    df.to_csv(csv_path, index=False)

    result = load_raw_data(csv_path)
    assert "extra_col" in result.columns
    for col in EXPECTED_COLUMNS:
        assert col in result.columns


def test_load_raw_data_error_message_names_missing_columns(tmp_path):
    """The ValueError message must mention the missing column name."""
    missing_col = "Torque [Nm]"
    cols = [c for c in EXPECTED_COLUMNS if c != missing_col]
    data = {col: [1] for col in cols}
    csv_path = tmp_path / "partial.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    # Escape special regex chars from the column name before matching
    import re as _re

    with pytest.raises(ValueError, match=_re.escape(missing_col)):
        load_raw_data(csv_path)
