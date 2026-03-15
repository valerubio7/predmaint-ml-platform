import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as st_components

API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="PredMaint Dashboard", page_icon="⚙️", layout="wide")

st.title("⚙️ PredMaint ML Platform")
st.caption("Predictive Maintenance — Real-time monitoring dashboard")

# Sidebar
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API URL", value=API_URL)

# Health Check
st.header("System Status")
try:
    response = httpx.get(f"{api_url}/health", timeout=3)
    if response.status_code == 200:
        st.success("API is online")
    else:
        st.error("API returned an error")
except Exception:
    st.error("API is offline — check the URL")

# Prediction
st.header("Real-time Prediction")
st.write("Enter sensor readings to get a failure prediction.")

col1, col2, col3 = st.columns(3)

with col1:
    air_temp = st.number_input("Air Temperature [K]", value=298.1, step=0.1)
    process_temp = st.number_input("Process Temperature [K]", value=308.6, step=0.1)
    rotational_speed = st.number_input("Rotational Speed [rpm]", value=1551, step=10)

with col2:
    torque = st.number_input("Torque [Nm]", value=42.8, step=0.1)
    tool_wear = st.number_input("Tool Wear [min]", value=0, step=1)

with col3:
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
    type_h = 1 if machine_type == "H" else 0
    type_l = 1 if machine_type == "L" else 0
    type_m = 1 if machine_type == "M" else 0

if st.button("Predict", type="primary"):
    payload = {
        "air_temperature": air_temp,
        "process_temperature": process_temp,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "type_h": type_h,
        "type_l": type_l,
        "type_m": type_m,
    }
    try:
        response = httpx.post(f"{api_url}/predict", json=payload, timeout=5)
        result = response.json()

        prob = result["failure_probability"]
        failure = result["failure_predicted"]

        if failure:
            st.error(f"⚠️ Failure predicted — probability: {prob:.1%}")
        else:
            st.success(f"✅ No failure predicted — probability: {prob:.1%}")

        st.metric("Failure Probability", f"{prob:.1%}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Dataset Overview
st.header("Dataset Overview")
data_path = Path("data/raw/ai4i2020.csv")

if data_path.exists():
    df = pd.read_csv(data_path)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Failure Rate", f"{df['Machine failure'].mean():.1%}")
    col3.metric("Features", "8")
    col4.metric("Model", "XGBoost")

    fig = px.histogram(
        df,
        x="Torque [Nm]",
        color="Machine failure",
        title="Torque Distribution by Failure",
        color_discrete_map={0: "steelblue", 1: "tomato"},
        barmode="overlay",
        opacity=0.7,
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.warning("Dataset not found at data/raw/ai4i2020.csv")


# Drift Reports
_REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "drift"
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def _list_drift_reports(reports_dir: Path) -> list[Path]:
    """Return HTML drift reports sorted newest-first."""
    if not reports_dir.exists():
        return []
    return sorted(reports_dir.glob("*.html"), reverse=True)


def _parse_report_label(path: Path) -> str:
    """Convert a report filename to a human-readable label.

    Example: drift_report_20260315T143000Z.html -> 2026-03-15 14:30:00 UTC
    Falls back to the raw filename if parsing fails.
    """
    stem = path.stem  # e.g. drift_report_20260315T143000Z
    raw_timestamp = stem.split("drift_report_")[-1]
    try:
        dt = datetime.strptime(raw_timestamp, _TIMESTAMP_FORMAT).replace(
            tzinfo=timezone.utc
        )
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return path.name


st.header("Drift Reports")

drift_reports = _list_drift_reports(_REPORTS_DIR)

if not drift_reports:
    st.info(
        "No drift reports found. Run `make drift` to generate one.",
        icon="ℹ️",
    )
else:
    report_labels = [_parse_report_label(p) for p in drift_reports]
    selected_label = st.selectbox(
        "Select report",
        options=report_labels,
        index=0,
    )
    selected_path = drift_reports[report_labels.index(selected_label)]

    st.caption(f"File: `{selected_path.name}`")
    st_components.html(
        selected_path.read_text(encoding="utf-8"),
        height=800,
        scrolling=True,
    )
