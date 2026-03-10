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


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Transform raw data into model-ready features and target."""
    df = df.drop(columns=DROP_COLUMNS)
    df = pd.get_dummies(df, columns=[CATEGORICAL_COLUMN], drop_first=False)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


if __name__ == "__main__":
    from data.ingest import load_raw_data

    df = load_raw_data()
    X, y = build_features(df)

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True).round(3)}")
    print(f"Feature columns:\n{list(X.columns)}")
