from pathlib import Path

import pandas as pd
import yaml

from data.loader import load_raw_data
from data.transformer import build_features

CONFIG_PATH = Path("configs/training.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load training configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Run full data pipeline: ingest → build features → save."""
    raw_path = Path(config["data"]["raw_path"])
    processed_path = Path(config["data"]["processed_path"])

    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw_path)
    X, y = build_features(df)

    output = X.copy()
    output["target"] = y
    output.to_parquet(processed_path, index=False)

    print("Pipeline complete.")
    print(f"Features: {X.shape[1]} columns, {X.shape[0]} rows")
    print(f"Saved to: {processed_path}")

    return X, y


if __name__ == "__main__":
    config = load_config()
    run_pipeline(config)
