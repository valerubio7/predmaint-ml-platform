import re

import pandas as pd

FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET_COLUMN = "Machine failure"
CATEGORICAL_COLUMN = "Type"
DROP_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace characters not allowed in XGBoost feature names ([ ] <)."""
    df.columns = [re.sub(r"[\[\]<]", "", col).strip() for col in df.columns]
    df.columns = [re.sub(r"\s+", "_", col) for col in df.columns]
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Transform raw data into model-ready features and target."""
    df = df.drop(columns=DROP_COLUMNS)
    df = pd.get_dummies(df, columns=[CATEGORICAL_COLUMN], drop_first=False)
    df = _sanitize_columns(df)

    target_col = re.sub(r"\s+", "_", TARGET_COLUMN)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


if __name__ == "__main__":
    from ingestion.ingest import load_raw_data

    df = load_raw_data()
    X, y = build_features(df)

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True).round(3)}")
    print(f"Feature columns:\n{list(X.columns)}")
