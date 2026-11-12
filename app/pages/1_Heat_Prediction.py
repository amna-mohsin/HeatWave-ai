import ui
import streamlit as st
import plotly.graph_objects as go
from src.predict import ModelNotAvailable, load_model, predict_live, predict_scenario

ui.setup("Heat Prediction")
st.title("Heat risk prediction")
tab_live, tab_manual = st.tabs(["Live Karachi data", "Manual scenario"])

try:
    _, meta = load_model()
except ModelNotAvailable as e:
    st.error(str(e)); st.stop()


def explain(res):
    ui.risk_card(res)
    c1, c2 = st.columns([1, 1])
    with c1:
        ui.precautions()
        st.caption(f"Model: {res['model']} · issue date {res['date']} · source: {res['source']}")
    with c2:
        fig = go.Figure(go.Bar(x=["Predicted probability", "Decision threshold"], y=[res["probability"], res["threshold"]],
                               marker_color=["#F97316", "#101B2D"]))
        fig.update_layout(height=280, yaxis=dict(range=[0, 1], title="probability"), margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
        ui.show(fig)
    st.info("The probability is the model's estimate, not a guaranteed risk. It is only as good as its training data (ERA5 grid + one airport station).")


with tab_live:
    st.write("Uses the last 14 days plus today from Open-Meteo to build the same features the model was trained on.")
    if st.button("Predict from live data", type="primary"):
        try:
            with st.spinner("Fetching weather…"):
                explain(predict_live())
        except Exception:
            st.error("Could not fetch live weather. Please try again or use the manual scenario.")

with tab_manual:
    st.write("Describe today's conditions. The recent past is assumed to be similar (steady-state scenario). Empty optional fields are imputed by the pipeline.")
    with st.form("manual"):
        c = st.columns(4)
        tmax = c[0].number_input("Max temperature (°C)", -10.0, 55.0, 38.0, 0.1)
        tmin = c[1].number_input("Min temperature (°C)", -20.0, 45.0, 28.0, 0.1)
        rh = c[2].number_input("Mean humidity (%)", 0.0, 100.0, 65.0, 1.0)
        dew = c[3].number_input("Mean dew point (°C)", -30.0, 35.0, 24.0, 0.1)
        c = st.columns(3)
        pres = c[0].number_input("Pressure (hPa)", 850.0, 1100.0, 1004.0, 0.5)
        wind = c[1].number_input("Max wind (km/h)", 0.0, 200.0, 20.0, 0.5)
        cloud = c[2].number_input("Cloud cover (%)", 0.0, 100.0, 20.0, 1.0)
        go_btn = st.form_submit_button("Predict", type="primary")
    if go_btn:
        if tmin > tmax:
            st.error("Minimum temperature cannot exceed maximum temperature.")
        elif dew > tmax:
            st.error("Dew point cannot exceed the maximum temperature.")
        else:
            explain(predict_scenario({"temperature_2m_max": tmax, "temperature_2m_min": tmin, "relative_humidity_2m_mean": rh,
                                      "dew_point_2m_mean": dew, "surface_pressure_mean": pres, "wind_speed_10m_max": wind,
                                      "cloud_cover_mean": cloud}))
