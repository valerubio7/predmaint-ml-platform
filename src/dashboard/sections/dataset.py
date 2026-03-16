import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_PLOTLY_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#94a3b8", "family": "Inter, sans-serif"},
    "xaxis": {"gridcolor": "#1e2333", "linecolor": "#2a3050"},
    "yaxis": {"gridcolor": "#1e2333", "linecolor": "#2a3050"},
    "margin": {"t": 40, "b": 40, "l": 40, "r": 20},
}


def render(df_raw: pd.DataFrame | None) -> None:
    st.markdown(
        '<div class="pm-section-header">Dataset Overview</div>',
        unsafe_allow_html=True,
    )

    if df_raw is None:
        _render_missing_dataset()
        return

    _render_metrics(df_raw)
    _render_charts(df_raw)


_MISSING_STYLE = (
    "padding:24px;background:#1a1d27;"
    "border:1px solid rgba(255,255,255,0.06);"
    "border-radius:10px;color:#475569;font-size:0.88rem;text-align:center;"
)


def _render_missing_dataset() -> None:
    st.markdown(
        f'<div style="{_MISSING_STYLE}">'
        "Dataset not found at <code>data/raw/ai4i2020.csv</code>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_metrics(df_raw: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df_raw):,}")
    col2.metric("Failure Rate", f"{df_raw['Machine failure'].mean():.1%}")
    col3.metric("Features", "8")
    col4.metric("Model", "XGBoost")


def _render_charts(df_raw: pd.DataFrame) -> None:
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.plotly_chart(_torque_histogram(df_raw), use_container_width=True)

    with chart_col2:
        st.plotly_chart(_failure_rate_by_type(df_raw), use_container_width=True)


def _torque_histogram(df_raw: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df_raw,
        x="Torque [Nm]",
        color="Machine failure",
        title="Torque Distribution by Failure",
        color_discrete_map={0: "#3b82f6", 1: "#ef4444"},
        barmode="overlay",
        opacity=0.75,
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#64748b", size=11),
        ),
    )
    return fig


def _failure_rate_by_type(df_raw: pd.DataFrame) -> go.Figure:
    failure_by_type = df_raw.groupby("Type")["Machine failure"].mean().reset_index()
    failure_by_type.columns = ["Machine Type", "Failure Rate"]
    fig = px.bar(
        failure_by_type,
        x="Machine Type",
        y="Failure Rate",
        title="Failure Rate by Machine Type",
        color="Failure Rate",
        color_continuous_scale=["#1e3a5f", "#3b82f6", "#ef4444"],
        text_auto=".1%",  # type: ignore[arg-type]
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT, coloraxis_showscale=False, yaxis_tickformat=".0%"
    )
    fig.update_traces(textfont_color="#e2e8f0", textposition="outside")
    return fig
