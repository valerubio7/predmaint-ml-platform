import streamlit as st


def render(
    api_url: str,
    mlflow_url: str,
    prefect_url: str,
    api_online: bool,
    mlflow_online: bool,
    prefect_online: bool,
) -> None:
    _render_service_bar(
        api_url, mlflow_url, prefect_url, api_online, mlflow_online, prefect_online
    )
    _render_status_indicators(api_online, mlflow_online, prefect_online)


def _dot_class(online: bool) -> str:
    return "pm-status-dot-online" if online else "pm-status-dot-offline"


def _render_service_bar(
    api_url: str,
    mlflow_url: str,
    prefect_url: str,
    api_online: bool,
    mlflow_online: bool,
    prefect_online: bool,
) -> None:
    api_dot = "pm-dot-online" if api_online else "pm-dot-offline"
    mlflow_dot = "pm-dot-online" if mlflow_online else "pm-dot-offline"
    prefect_dot = "pm-dot-online" if prefect_online else "pm-dot-offline"

    st.markdown(
        f"""
        <div class="pm-service-bar">
            <a class="pm-service-link" href="{mlflow_url}" target="_blank">
                <span class="pm-service-link-dot {mlflow_dot}"></span>
                MLflow UI
            </a>
            <a class="pm-service-link" href="{prefect_url}" target="_blank">
                <span class="pm-service-link-dot {prefect_dot}"></span>
                Prefect Dashboard
            </a>
            <a class="pm-service-link" href="{api_url}/docs" target="_blank">
                <span class="pm-service-link-dot {api_dot}"></span>
                API Docs (Swagger)
            </a>
            <a class="pm-service-link" href="{api_url}/redoc" target="_blank">
                <span class="pm-service-link-dot {api_dot}"></span>
                API Docs (ReDoc)
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status_indicators(
    api_online: bool,
    mlflow_online: bool,
    prefect_online: bool,
) -> None:
    st.markdown(
        '<div class="pm-section-header">System Status</div>', unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        _status_row("Prediction API", api_online)
    with col2:
        _status_row("MLflow Tracking", mlflow_online)
    with col3:
        _status_row("Prefect Orchestrator", prefect_online)


def _status_row(name: str, online: bool) -> None:
    dot = _dot_class(online)
    label = "Online" if online else "Offline"
    st.markdown(
        f"""<div class="pm-status-row">
            <div class="pm-status-dot {dot}"></div>
            <span class="pm-status-name">{name}</span>
            <span class="pm-status-label">{label}</span>
        </div>""",
        unsafe_allow_html=True,
    )
