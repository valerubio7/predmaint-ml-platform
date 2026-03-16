from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as st_components

_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def render(reports_dir: Path) -> None:
    st.markdown(
        '<div class="pm-section-header">Data Drift Reports</div>',
        unsafe_allow_html=True,
    )

    reports = _list_reports(reports_dir)

    if not reports:
        _render_no_reports()
        return

    _render_report_viewer(reports)


def _list_reports(reports_dir: Path) -> list[Path]:
    if not reports_dir.exists():
        return []
    return sorted(reports_dir.glob("*.html"), reverse=True)


def _parse_report_label(path: Path) -> str:
    stem = path.stem
    raw_timestamp = stem.split("drift_report_")[-1]
    try:
        dt = datetime.strptime(raw_timestamp, _TIMESTAMP_FORMAT).replace(
            tzinfo=timezone.utc
        )
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return path.name


def _render_no_reports() -> None:
    st.markdown(
        """
        <div style="
            padding:24px;background:#1a1d27;
            border:1px dashed rgba(245,158,11,0.2);
            border-radius:10px;color:#475569;
            font-size:0.88rem;text-align:center;
        ">
            No drift reports found — run <code style="color:#f59e0b;">make drift</code>
            to generate an Evidently report.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_report_viewer(reports: list[Path]) -> None:
    labels = [_parse_report_label(p) for p in reports]
    selected_label = st.selectbox(
        "Select report",
        options=labels,
        index=0,
        label_visibility="collapsed",
    )
    selected_path = reports[labels.index(selected_label)]

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    margin-bottom:12px;">
            <span style="font-size:0.78rem;color:#475569;">
                <code style="color:#64748b;">{selected_path.name}</code>
            </span>
            <span style="font-size:0.72rem;color:#f59e0b;
                font-family:'JetBrains Mono',monospace;">
                {selected_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st_components.html(
        selected_path.read_text(encoding="utf-8"),
        height=820,
        scrolling=True,
    )
