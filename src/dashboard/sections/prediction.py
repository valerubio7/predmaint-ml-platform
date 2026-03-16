import httpx
import streamlit as st


def render(api_url: str) -> None:
    st.markdown(
        '<div class="pm-section-header">Real-time Inference</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        air_temp = st.number_input("Air Temperature [K]", value=298.1, step=0.1)
        process_temp = st.number_input("Process Temperature [K]", value=308.6, step=0.1)

    with col2:
        rotational_speed = st.number_input(
            "Rotational Speed [rpm]", value=1551, step=10
        )
        torque = st.number_input("Torque [Nm]", value=42.8, step=0.1)

    with col3:
        tool_wear = st.number_input("Tool Wear [min]", value=0, step=1)
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

    with col4:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button(
            "Run Prediction", type="primary", use_container_width=True
        )

    if predict_clicked:
        _run_prediction(
            api_url=api_url,
            air_temp=air_temp,
            process_temp=process_temp,
            rotational_speed=rotational_speed,
            torque=torque,
            tool_wear=tool_wear,
            machine_type=machine_type,
        )


def _run_prediction(
    api_url: str,
    air_temp: float,
    process_temp: float,
    rotational_speed: int,
    torque: float,
    tool_wear: int,
    machine_type: str,
) -> None:
    payload = {
        "air_temperature": air_temp,
        "process_temperature": process_temp,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "type_h": 1 if machine_type == "H" else 0,
        "type_l": 1 if machine_type == "L" else 0,
        "type_m": 1 if machine_type == "M" else 0,
    }
    try:
        response = httpx.post(f"{api_url}/predict", json=payload, timeout=5)
        result = response.json()
        _render_prediction_result(
            failure_probability=result["failure_probability"],
            failure_predicted=result["failure_predicted"],
        )
    except Exception as exc:
        st.error(f"Prediction request failed: {exc}")


def _render_prediction_result(
    failure_probability: float,
    failure_predicted: bool,
) -> None:
    result_class = "pm-result-failure" if failure_predicted else "pm-result-ok"
    result_icon = "⚠️" if failure_predicted else "✓"
    result_title = "Failure predicted" if failure_predicted else "No failure predicted"
    prob_color = "#ef4444" if failure_predicted else "#10b981"

    st.markdown(
        f"""
        <div class="{result_class}" style="margin-top:16px;">
            <div class="pm-result-title" style="color:{prob_color};">
                {result_icon} {result_title}
            </div>
            <div class="pm-result-prob" style="color:{prob_color};">
                {failure_probability:.1%}
            </div>
            <div style="font-size:0.78rem;color:#475569;margin-top:6px;">
                failure probability
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
