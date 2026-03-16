import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/training.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
