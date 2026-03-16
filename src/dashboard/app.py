import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from dashboard.sections import dataset, drift, hero, prediction, status

API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")
PREFECT_URL = os.getenv("PREFECT_URL", "http://localhost:4200")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_PATH = _PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
_REPORTS_DIR = _PROJECT_ROOT / "reports" / "drift"

st.set_page_config(
    page_title="PredMaint — ML Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_CSS_PATH = Path(__file__).parent / "styles.css"
st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


def _check_service(url: str, timeout: float = 2.5) -> bool:
    try:
        httpx.get(url, timeout=timeout)
        return True
    except Exception:
        return False


def _load_dataset() -> pd.DataFrame | None:
    return pd.read_csv(_DATA_PATH) if _DATA_PATH.exists() else None


def main() -> None:
    df_raw = _load_dataset()

    hero.render(df_raw)

    api_online = _check_service(f"{API_URL}/health")
    mlflow_online = _check_service(MLFLOW_URL)
    prefect_online = _check_service(f"{PREFECT_URL}/api/health")

    status.render(
        api_url=API_URL,
        mlflow_url=MLFLOW_URL,
        prefect_url=PREFECT_URL,
        api_online=api_online,
        mlflow_online=mlflow_online,
        prefect_online=prefect_online,
    )

    prediction.render(api_url=API_URL)
    dataset.render(df_raw)
    drift.render(reports_dir=_REPORTS_DIR)


main()
