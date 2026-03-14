from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("data/raw/ai4i2020.csv")

EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Data loaded: {df.shape}")
    print(df.dtypes)
