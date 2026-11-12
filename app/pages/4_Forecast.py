import ui
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import HI_THRESHOLD_C
from src.predict import forecast_heat_index

ui.setup("Forecast")
st.title("Seven-day forecast")


@st.cache_data(ttl=1800, show_spinner="Fetching forecast…")
def get():
    return forecast_heat_index(7)


try:
    fc = get()
except Exception:
    st.error("Could not fetch the forecast from Open-Meteo. Please retry shortly."); st.stop()

fig = make_subplots(specs=[[{"secondary_y": True}]])
x = fc["date"].dt.strftime("%a %d %b")
fig.add_trace(go.Scatter(x=x, y=fc["temperature_2m_max"], name="Max temp (°C)", line=dict(color="#F97316")))
fig.add_trace(go.Scatter(x=x, y=fc["heat_index_c"], name="Heat index (°C)", line=dict(color="#E5484D", width=3)))
fig.add_trace(go.Bar(x=x, y=fc["relative_humidity_2m_mean"], name="Humidity (%)", marker_color="rgba(56,189,248,.35)"), secondary_y=True)
fig.add_hline(y=HI_THRESHOLD_C, line_dash="dash", line_color="#E5484D")
fig.update_layout(height=400, plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
ui.show(fig)
t = fc[["date", "temperature_2m_max", "temperature_2m_min", "relative_humidity_2m_mean", "heat_index_c", "above_threshold"]].copy()
t["date"] = t["date"].dt.strftime("%a %d %b")
t.columns = ["Day", "Max °C", "Min °C", "Humidity %", "Heat index °C", f"≥ {HI_THRESHOLD_C} °C?"]
st.dataframe(t.round(1))
st.caption("Heat index is computed from forecast max temperature and mean dew point (same formula as the labels). Source: Open-Meteo forecast API.")
