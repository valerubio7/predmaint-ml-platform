import logging
from pathlib import Path

import pandas as pd

from core.features import build_features
from core.loader import load_raw_data

logger = logging.getLogger(__name__)


def load_features(raw_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw data and return (X, y) without writing anything to disk."""
    df = load_raw_data(raw_path)
    return build_features(df)


def _save_features(X: pd.DataFrame, y: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    X.assign(target=y).to_parquet(path, index=False)
    logger.info("Saved to: %s", path)


def run_pipeline(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    raw_path = Path(config["data"]["raw_path"])
    processed_path = Path(config["data"]["processed_path"])

    df = load_raw_data(raw_path)
    X, y = build_features(df)

    _save_features(X, y, processed_path)

    logger.info(
        "Pipeline complete. Features: %d columns, %d rows", X.shape[1], X.shape[0]
    )

    return X, y


if __name__ == "__main__":
    from core.config import load_config

    config = load_config()
    run_pipeline(config)
