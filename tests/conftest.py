"""Shared fixtures and data builders for all test modules."""

from pathlib import Path

import pandas as pd
import pytest

from core.loader import EXPECTED_COLUMNS

# ---------------------------------------------------------------------------
# Raw data builders
# ---------------------------------------------------------------------------

ALL_TYPES = ["L", "M", "H"]


def make_raw_df(rows: int = 20, types: list[str] | None = None) -> pd.DataFrame:
    """Return a synthetic raw DataFrame matching the AI4I schema."""
    if types is None:
        third = rows // 3
        types = ["L"] * third + ["M"] * third + ["H"] * (rows - 2 * third)

    return pd.DataFrame(
        {
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
    )


def make_raw_csv(tmp_path: Path, rows: int = 20) -> Path:
    """Write a minimal valid raw CSV and return its path."""
    csv_path = tmp_path / "raw.csv"
    make_raw_df(rows=rows).to_csv(csv_path, index=False)
    return csv_path


def make_valid_csv(tmp_path: Path, rows: int = 5) -> Path:
    """Write a CSV with all EXPECTED_COLUMNS and return its path."""
    data: dict[str, list] = {col: [0] * rows for col in EXPECTED_COLUMNS}
    data["Type"] = ["M"] * rows
    data["Product ID"] = [f"M{i}" for i in range(rows)]
    csv_path = tmp_path / "valid.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_df_small() -> pd.DataFrame:
    """Synthetic raw DataFrame with 30 rows covering all three Type values."""
    return make_raw_df(rows=30)


@pytest.fixture
def single_type_df():
    """Factory fixture: returns a raw DataFrame where every row has the given Type."""

    def _factory(type_val: str) -> pd.DataFrame:
        return make_raw_df(rows=10, types=[type_val] * 10)

    return _factory
