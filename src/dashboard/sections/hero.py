import pandas as pd
import streamlit as st


def render(df_raw: pd.DataFrame | None) -> None:
    total_records = f"{len(df_raw):,}" if df_raw is not None else "—"
    failure_rate = (
        f"{df_raw['Machine failure'].mean():.1%}" if df_raw is not None else "—"
    )

    st.markdown(
        f"""
        <div class="pm-hero">
            <div class="pm-hero-title">Predictive Maintenance Platform</div>
            <div class="pm-hero-subtitle">
                End-to-end ML pipeline — from raw sensor data to production inference
            </div>
            <div class="pm-stack">
                <span class="pm-badge pm-badge-accent">XGBoost</span>
                <span class="pm-badge">MLflow</span>
                <span class="pm-badge">Prefect</span>
                <span class="pm-badge">Evidently</span>
                <span class="pm-badge">FastAPI</span>
                <span class="pm-badge">Streamlit</span>
                <span class="pm-badge">Docker</span>
            </div>
            <div class="pm-hero-stats">
                <div>
                    <div class="pm-hero-stat-label">Dataset records</div>
                    <div class="pm-hero-stat-value">{total_records}</div>
                    <div class="pm-hero-stat-delta">AI4I 2020</div>
                </div>
                <div>
                    <div class="pm-hero-stat-label">Failure rate</div>
                    <div class="pm-hero-stat-value">{failure_rate}</div>
                    <div class="pm-hero-stat-delta">class imbalance handled</div>
                </div>
                <div>
                    <div class="pm-hero-stat-label">Features</div>
                    <div class="pm-hero-stat-value">8</div>
                    <div class="pm-hero-stat-delta">sensor + machine type</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
