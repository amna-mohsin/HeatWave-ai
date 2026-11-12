import ui
import streamlit as st
import plotly.express as px
from src.config import HI_THRESHOLD_C, TARGET, PROC_DIR
from src.data_loader import load_dataset

ui.setup("Weather Analytics")
st.title("Weather analytics")


@st.cache_data
def data():
    d = load_dataset(); d["year"] = d["date"].dt.year; d["month"] = d["date"].dt.month; return d


if not (PROC_DIR / "heatwave_dataset.csv").exists():
    st.error("Dataset not found. Run the data pipeline first."); st.stop()
df = data()
y0, y1 = st.slider("Years", int(df.year.min()), int(df.year.max()), (int(df.year.min()), int(df.year.max())))
d = df[(df.year >= y0) & (df.year <= y1)]
c1, c2 = st.columns(2)
with c1:
    t = d.groupby("year")["temperature_2m_max"].mean().reset_index()
    ui.show(px.line(t, x="year", y="temperature_2m_max", title="Mean daily max temperature per year (°C)", color_discrete_sequence=["#F97316"]))
    st.caption("Each point is a year's average daily high. The line helps show long-term changes, not the temperature on a specific day.")
with c2:
    m = d.groupby("month")[TARGET].mean().mul(100).reset_index()
    ui.show(px.bar(m, x="month", y=TARGET, title="Days labelled dangerous, by month (%)", color_discrete_sequence=["#E5484D"]))
    st.caption("Taller bars mark months when dangerous heat labels happened more often in this dataset.")
c3, c4 = st.columns(2)
with c3:
    f = px.box(d, x="month", y="heat_index_era", title="Heat index by month (°C, ERA5-derived)", color_discrete_sequence=["#F97316"])
    f.add_hline(y=HI_THRESHOLD_C, line_dash="dash", line_color="#E5484D"); ui.show(f)
    st.caption("Boxes show the spread of estimated daily heat index by month; the dashed line is the project's 40.6 °C threshold.")
with c4:
    s = d.sample(min(len(d), 3000), random_state=1); s["label"] = s[TARGET].map({0: "Safe", 1: "Dangerous"})
    ui.show(px.scatter(s, x="temperature_2m_max", y="relative_humidity_2m_mean", color="label", opacity=.6,
                       color_discrete_map={"Safe": "#38BDF8", "Dangerous": "#E5484D"}, title="Temperature vs humidity"))
    st.caption("Each dot is a day. Red dots are dangerous labels; their position shows the day's high temperature and humidity.")
st.caption(f"{len(d):,} labelled days · {d.date.min().date()} to {d.date.max().date()}")
