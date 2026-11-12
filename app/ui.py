"""Shared Streamlit styling + helpers. Import this FIRST in every page (it fixes sys.path)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Plus+Jakarta+Sans:wght@600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { color:#12213A; font-family:'Plus Jakarta Sans',sans-serif !important; letter-spacing:-0.025em; }
[data-testid="stAppViewContainer"] { background:#F3F6FB; }
[data-testid="stHeader"] { background:rgba(243,246,251,.88); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0C172A 0%,#14243D 100%); border-right:1px solid #20344F; }
[data-testid="stSidebar"] * { color:#E2E8F0 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#A8B6C9 !important; }
[data-testid="stSidebarNav"] { padding-top:.35rem; }
[data-testid="stSidebarNav"]::before {
  content:"HEATWAVE AI\\A KARACHI\\A WEATHER INTELLIGENCE";
  white-space:pre-wrap;
  overflow-wrap:anywhere;
  display:block;
  color:#F8FAFC;
  font:700 12px 'Plus Jakarta Sans',sans-serif;
  letter-spacing:.09em;
  line-height:1.65;
  padding:0 10px 13px;
  margin:0 0 11px;
  border-bottom:1px solid #2B3D55;
}
[data-testid="stSidebarNav"] a { border-radius:10px; margin:.16rem 0; padding:.48rem .7rem; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background:#253B56; border-left:3px solid #FB923C; }
[data-testid="stSidebarNav"] a:hover { background:#20344F; }
.block-container { padding-top:1.8rem; padding-bottom:3rem; max-width:1240px; }
.hw-sidebar-panel { background:rgba(255,255,255,.055); border:1px solid #2B3D55; border-radius:14px; margin-top:1.5rem; padding:15px 14px; }
.hw-sidebar-kicker { color:#FB923C; font-size:10px; font-weight:700; letter-spacing:.13em; text-transform:uppercase; }
.hw-sidebar-title { color:#F8FAFC; font:700 15px 'Plus Jakarta Sans',sans-serif; margin:.35rem 0 .25rem; }
.hw-sidebar-copy { color:#C2CDDA; font-size:11px; line-height:1.55; }
.hw-sidebar-meta { border-top:1px solid #2B3D55; color:#C2CDDA; font-size:10px; margin-top:12px; padding-top:10px; }
.hw-topline { align-items:center; background:#fff; border:1px solid #D7E0EB; border-radius:999px; box-shadow:0 3px 12px rgba(15,35,65,.045); color:#111827 !important; display:inline-flex; font-size:11px; font-weight:800; gap:9px; letter-spacing:.1em; margin:.1rem 0 1.15rem; padding:8px 13px; text-transform:uppercase; }
.hw-topline, .hw-topline * { color:#111827 !important; opacity:1 !important; }
.hw-dot { background:#16834A; border-radius:50%; box-shadow:0 0 0 4px rgba(22,131,74,.12); display:inline-block; flex:0 0 7px; height:7px; width:7px; }
.hw-card { background:#fff; border:1px solid #E3EAF3; border-radius:16px; box-shadow:0 5px 18px rgba(15,35,65,.045); padding:18px 20px; }
.hw-card .k { color:#64748B; font-size:12px; font-weight:600; letter-spacing:.035em; }
.hw-card .v { color:#14243D; font:800 29px 'Plus Jakarta Sans',sans-serif; font-variant-numeric:tabular-nums; margin-top:5px; }
.hw-heat .v { color:#EA580C; }
.hw-risk { border:1px solid rgba(255,255,255,.2); border-radius:18px; box-shadow:0 12px 26px rgba(15,35,65,.13); color:#fff; min-height:160px; padding:23px 26px; }
.hw-risk.danger { background:linear-gradient(120deg,#C9343D,#E5484D 72%,#F16A55); }
.hw-risk.elevated { background:linear-gradient(120deg,#D65A16,#F97316 72%,#FB923C); }
.hw-risk.safe { background:linear-gradient(120deg,#138044,#22A45A 72%,#31B76D); }
.hw-risk .p { font:800 54px 'Plus Jakarta Sans',sans-serif; font-variant-numeric:tabular-nums; line-height:1.08; margin:.35rem 0; }
.hw-risk .s { font-size:12px; opacity:.92; }
[data-testid="stMetric"] { background:#fff; border:1px solid #E3EAF3; border-radius:14px; box-shadow:0 5px 18px rgba(15,35,65,.04); padding:14px 16px; }
[data-testid="stMetricLabel"] { color:#64748B; }
[data-testid="stMetricValue"] { color:#14243D; font-family:'Plus Jakarta Sans',sans-serif; }
[data-testid="stTabs"] button { font-weight:600; }
[data-testid="stForm"] { background:#fff; border:1px solid #E3EAF3; border-radius:16px; padding:18px; }
div.stButton > button, div.stFormSubmitButton > button { border-radius:10px; font-weight:600; min-height:2.7rem; }
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] { box-shadow:0 5px 12px rgba(249,115,22,.2); }
[data-testid="stDataFrame"] { border:1px solid #E3EAF3; border-radius:12px; overflow:hidden; }
@media (max-width: 700px) {
  .block-container { padding-left:1rem; padding-right:1rem; padding-top:1rem; }
  .hw-risk .p { font-size:44px; }
}
</style>
"""


def setup(title: str, icon: str = "🌡️"):
    st.set_page_config(page_title=f"{title} | HeatWave AI", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(
            '<div class="hw-sidebar-panel">'
            '<div class="hw-sidebar-kicker">Forecast horizon</div>'
            '<div class="hw-sidebar-title">Heat risk · 5–7 days ahead</div>'
            '<div class="hw-sidebar-copy">A machine-learning study of historical Karachi weather, humidity, and heat-index patterns.</div>'
            '<div class="hw-sidebar-meta">KARACHI&nbsp;&nbsp;·&nbsp;&nbsp;WEEK-AHEAD OUTLOOK</div>'
            '</div>',
            unsafe_allow_html=True,   
        )
    st.markdown(
        '<div class="hw-topline"><span class="hw-dot"></span>'
        'Karachi heat-risk intelligence&nbsp;&nbsp;·&nbsp;&nbsp;historical + live weather</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, heat=False):
    st.markdown(f'<div class="hw-card {"hw-heat" if heat else ""}"><div class="k">{label}</div><div class="v">{value}</div></div>',
                unsafe_allow_html=True)


def risk_card(res: dict):
    cls = "danger" if res["prediction"] == 1 else ("elevated" if res["risk_band"] == "Elevated" else "safe")
    st.markdown(f'<div class="hw-risk {cls}"><div>{res["risk_band"].upper()}</div>'
                f'<div class="p">{res["probability"]*100:.1f}%</div>'
                f'<div class="s">Predicted probability for days 5–7 (not a guarantee) · decision threshold {res["threshold"]*100:.1f}%</div></div>',
                unsafe_allow_html=True)


def show(fig):
    """Render a polished Plotly chart without zoom, pan, hover, or edit controls."""
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color="#526176"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    config = {
        "staticPlot": True,
        "displayModeBar": False,
        "scrollZoom": False,
        "responsive": True,
    }
    try:
        st.plotly_chart(fig, width="stretch", config=config)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=config)


def precautions():
    st.markdown("**Recommended precautions**\n- Avoid prolonged outdoor exposure in the afternoon.\n- Stay hydrated; prefer shaded or cooled spaces.\n"
                "- Check on elderly neighbours and outdoor workers.\n- Monitor official PMD advisories.")
