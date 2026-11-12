import ui
import streamlit as st
import plotly.graph_objects as go
from src.config import HI_THRESHOLD_C
from src.predict import ModelNotAvailable, forecast_heat_index, predict_live

ui.setup("Overview")
st.title("HeatWave AI")
st.caption("Week-ahead dangerous heat warning for Karachi · ML classification on ERA5 + NOAA GSOD data")


@st.cache_data(ttl=1800, show_spinner="Fetching live Karachi weather…")
def live_bundle():
    return predict_live(), forecast_heat_index(7)


try:
    res, fc = live_bundle()
except ModelNotAvailable as e:
    st.error(str(e)); st.stop()
except Exception:
    st.warning("Live weather could not be fetched right now (Open-Meteo unreachable). Try the manual scenario on the Heat Prediction page.")
    st.stop()

left, right = st.columns([1.2, 2])
with left:
    ui.risk_card(res)
    st.caption(f"Issue date {res['date']} · model: {res['model']} · source: {res['source']}")
with right:
    today = fc.iloc[0]
    c1, c2, c3 = st.columns(3)
    with c1: ui.metric_card("Max temp today", f"{today['temperature_2m_max']:.1f} °C", heat=True)
    with c2: ui.metric_card("Mean humidity", f"{today['relative_humidity_2m_mean']:.0f} %")
    with c3: ui.metric_card("Peak heat index", f"{today['heat_index_c']:.1f} °C", heat=True)

st.subheader("Forecast heat index, next 7 days")
colors = ["#E5484D" if x else "#F97316" for x in fc["above_threshold"]]
fig = go.Figure(go.Bar(x=fc["date"].dt.strftime("%a %d %b"), y=fc["heat_index_c"].round(1), marker_color=colors))
fig.add_hline(y=HI_THRESHOLD_C, line_dash="dash", line_color="#E5484D", annotation_text=f"{HI_THRESHOLD_C} °C threshold")
fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="°C", plot_bgcolor="white")
ui.show(fig)
st.caption("Bars come from the weather-model forecast itself (a cross-check). The ML probability above is a separate, learned estimate.")
