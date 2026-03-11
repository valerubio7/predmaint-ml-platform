"""Tests for src/data/transformer.py."""

import re

import pandas as pd
import pytest

from data.transformer import (
    DROP_COLUMNS,
    _sanitize_columns,
    build_features,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_df(rows: int = 20, types: list[str] | None = None) -> pd.DataFrame:
    """Return a minimal synthetic raw DataFrame matching the real schema."""
    if types is None:
        types = (
            ["L"] * (rows // 3) + ["M"] * (rows // 3) + ["H"] * (rows - 2 * (rows // 3))
        )
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
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# _sanitize_columns tests
# ---------------------------------------------------------------------------


def test_sanitize_columns_removes_brackets():
    """Square brackets must be stripped from column names."""
    df = pd.DataFrame({"Air temperature [K]": [1], "Rotational speed [rpm]": [2]})
    result = _sanitize_columns(df)
    assert "Air_temperature_K" in result.columns
    assert "Rotational_speed_rpm" in result.columns


def test_sanitize_columns_removes_angle_brackets():
    """Only the less-than sign (<) must be stripped; > is not in the regex."""
    df = pd.DataFrame({"Feature<val": [1]})
    result = _sanitize_columns(df)
    # _sanitize_columns strips [ ] and < only (per XGBoost restriction)
    assert "Feature_val" not in result.columns  # spaces → underscores, no spaces here
    for col in result.columns:
        assert "<" not in col


def test_sanitize_columns_replaces_spaces_with_underscores():
    """Spaces must be replaced with underscores."""
    df = pd.DataFrame({"my feature col": [1]})
    result = _sanitize_columns(df)
    assert "my_feature_col" in result.columns


def test_sanitize_columns_strips_leading_trailing_spaces():
    """Leading/trailing spaces must be stripped before underscore conversion."""
    df = pd.DataFrame({" col ": [1]})
    result = _sanitize_columns(df)
    assert "col" in result.columns


def test_sanitize_columns_leaves_clean_names_unchanged():
    """Column names with no special chars must remain the same."""
    df = pd.DataFrame({"clean_name": [1], "another": [2]})
    result = _sanitize_columns(df)
    assert list(result.columns) == ["clean_name", "another"]


def test_sanitize_columns_no_xgboost_illegal_chars():
    """Result column names must not contain [ ] or <."""
    df = _make_raw_df()
    result = _sanitize_columns(df)
    for col in result.columns:
        assert not re.search(r"[\[\]<]", col), f"Illegal XGBoost char in: {col}"


# ---------------------------------------------------------------------------
# build_features happy-path tests
# ---------------------------------------------------------------------------


def test_build_features_returns_tuple(raw_data_small):
    """build_features must return a (DataFrame, Series) tuple."""
    X, y = build_features(raw_data_small)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_build_features_drops_unused_columns(raw_data_small):
    """DROP_COLUMNS must not appear in the output feature set."""
    X, _ = build_features(raw_data_small)
    for col in DROP_COLUMNS:
        assert col not in X.columns, f"Dropped column still present: {col}"


def test_build_features_no_raw_type_column(raw_data_small):
    """Original 'Type' column must be gone after one-hot encoding."""
    X, _ = build_features(raw_data_small)
    assert "Type" not in X.columns


def test_build_features_ohe_type_columns_present(raw_data_small):
    """One-hot encoded Type columns must exist in output."""
    X, _ = build_features(raw_data_small)
    assert "Type_H" in X.columns
    assert "Type_L" in X.columns
    assert "Type_M" in X.columns


def test_build_features_correct_column_count(raw_data_small):
    """Feature matrix must have exactly 8 columns for the AI4I schema."""
    X, _ = build_features(raw_data_small)
    # 5 numeric + 3 OHE (H/L/M) = 8
    assert X.shape[1] == 8


def test_build_features_target_not_in_X(raw_data_small):
    """Target column (Machine_failure) must not appear in X."""
    X, _ = build_features(raw_data_small)
    assert "Machine_failure" not in X.columns
    assert "Machine failure" not in X.columns


def test_build_features_target_is_series(raw_data_small):
    """y must be a pd.Series."""
    _, y = build_features(raw_data_small)
    assert isinstance(y, pd.Series)


def test_build_features_target_binary_values(raw_data_small):
    """All target values must be in {0, 1}."""
    _, y = build_features(raw_data_small)
    assert set(y.unique()).issubset({0, 1})


def test_build_features_no_nulls(raw_data_small):
    """Feature matrix must have zero missing values."""
    X, _ = build_features(raw_data_small)
    assert X.isnull().sum().sum() == 0


def test_build_features_row_count_preserved(raw_data_small):
    """Number of rows must be the same as the input."""
    n_rows = len(raw_data_small)
    X, y = build_features(raw_data_small)
    assert len(X) == n_rows
    assert len(y) == n_rows


def test_build_features_column_names_xgboost_safe(raw_data_small):
    """Feature column names must not contain [ ] < (XGBoost requirement)."""
    X, _ = build_features(raw_data_small)
    for col in X.columns:
        assert not re.search(r"[\[\]<]", col), f"Unsafe column name: {col}"


def test_build_features_ohe_values_are_boolean_int(raw_data_small):
    """OHE columns must contain only 0s and 1s (True/False)."""
    X, _ = build_features(raw_data_small)
    for col in ("Type_H", "Type_L", "Type_M"):
        assert set(X[col].unique()).issubset({0, 1, True, False})


def test_build_features_numeric_columns_present(raw_data_small):
    """All sanitised numeric sensor columns must be in the output."""
    expected_numeric = [
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min",
    ]
    X, _ = build_features(raw_data_small)
    for col in expected_numeric:
        assert col in X.columns, f"Numeric column missing: {col}"


def test_build_features_type_only_l(tmp_df_type):
    """When all rows have Type=L, Type_H and Type_M must be all zeros."""
    df = tmp_df_type("L")
    X, _ = build_features(df)
    assert "Type_L" in X.columns
    assert X["Type_L"].all()  # all L rows => Type_L == 1
    assert (X.get("Type_H", pd.Series([0])) == 0).all()
    assert (X.get("Type_M", pd.Series([0])) == 0).all()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_data_small():
    """Small synthetic raw DataFrame (30 rows) for fast unit tests."""
    return _make_raw_df(rows=30)


@pytest.fixture
def tmp_df_type():
    """Factory fixture: returns a raw DF where all rows have the given Type."""

    def _factory(type_val: str) -> pd.DataFrame:
        return _make_raw_df(rows=10, types=[type_val] * 10)

    return _factory
