import logging
from pathlib import Path

import pandas as pd
import yaml

from data.loader import load_raw_data
from data.transformer import build_features

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/training.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


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
    config = load_config()
    run_pipeline(config)
