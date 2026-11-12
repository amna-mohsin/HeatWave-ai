import ui
import streamlit as st

ui.setup("About")
st.title("About HeatWave AI")
st.markdown("""
**Problem.** Karachi's hot, humid summers can reach dangerous heat-index levels. HeatWave AI estimates the probability that
the heat index reaches **≥ 40.6 °C (105 °F)** on any of days **5–7** after the issue day. The threshold is *this project's choice*,
not an official NOAA category boundary.

**Data (two public sources).**
- Open-Meteo Historical Weather API (ERA5 reanalysis) – input features – licence CC BY 4.0.
- NOAA NCEI Global Summary of the Day, Karachi Jinnah Intl Airport (41780099999) – target labels – NOAA open data (free to use, citation required).

**Method.** Cleaning → merge on date → feature engineering (rolling means, heat index, seasonality) → chronological 70/15/15
split with a 7-day purge → sklearn Pipeline (impute, log1p, scale, one-hot) → Dummy baseline + six classifiers:
Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Gradient Boosting, and SVM → 5-fold time-series GridSearchCV (PR-AUC)
→ threshold chosen on validation → single test evaluation.

**Limitations.** ERA5 is a gridded reanalysis, the live inputs come from a different product (Open-Meteo forecast API's recent days),
and labels come from a single airport station. Probabilities are not guaranteed real-world risks. This is a student project, not an
official warning system – follow Pakistan Meteorological Department advisories.

**Team.** *Add your names and roll numbers here.*
""")
