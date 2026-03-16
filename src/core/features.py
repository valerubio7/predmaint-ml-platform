import re

import pandas as pd

TARGET_COLUMN = "Machine failure"
CATEGORICAL_COLUMN = "Type"
DROP_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

_XGBOOST_ILLEGAL_CHARS = re.compile(r"[\[\]<]")
_WHITESPACE = re.compile(r"\s+")


def _sanitize_column_name(name: str) -> str:
    cleaned = _XGBOOST_ILLEGAL_CHARS.sub("", name).strip()
    return _WHITESPACE.sub("_", cleaned)


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=_sanitize_column_name)


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=[CATEGORICAL_COLUMN], drop_first=False)


def _split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target_col = _sanitize_column_name(TARGET_COLUMN)
    return df.drop(columns=[target_col]), df[target_col]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    transformed = (
        df.drop(columns=DROP_COLUMNS).pipe(_encode_categoricals).pipe(_sanitize_columns)
    )
    return _split_features_and_target(transformed)


if __name__ == "__main__":
    from core.loader import load_raw_data

    df = load_raw_data()
    X, y = build_features(df)

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True).round(3)}")
    print(f"Feature columns:\n{list(X.columns)}")
