---
license: other
library_name: scikit-learn
tags: [tabular-classification, weather, heat-index, karachi, scikit-learn]
---
# HeatWave AI – Karachi dangerous-heat classifier

**Task:** binary classification – will station heat index reach ≥ 40.6 °C (105 °F) on any of days 5–7 after the issue day?
**Model:** see `model_metadata.json` (`model`, `best_params`, `decision_threshold`). The file `heatwave_pipeline.joblib` is the *entire* scikit-learn Pipeline (imputation, scaling/encoding, classifier).
Candidate comparison includes a prior Dummy baseline, a persistence baseline (today's heat index only) and four classifiers: Logistic Regression, Random Forest, Gradient Boosting, and SVM. Random Forest depth is capped to discourage overly complex trees.

## Usage
```python
import joblib, pandas as pd
pipe = joblib.load("heatwave_pipeline.joblib")      # only load files you trust (pickle)
proba = pipe.predict_proba(features_df)[:, 1]       # features_df: the 29 columns in model_metadata.json["features"]
danger = proba >= <decision_threshold from metadata>
```
Features must be built with `src/feature_engineering.build_features` from the GitHub repo.

## Data
Open-Meteo ERA5 (CC BY 4.0) for features; NOAA GSOD Karachi airport (NOAA open data, cite NCEI) for labels. Period and download date: `model_metadata.json`.
The model artifact does not declare a separate permissive license. Follow each source's applicable terms and attribution requirements; see `data/DATA_SOURCES.md`.

## Evaluation
Chronological train/validation/test split; test metrics in `model_metadata.json` (`test_metrics_tuned_threshold`). The selected model's train/validation PR-AUC difference is shown in its model insights page as an overfitting diagnostic; no result proves that a model is perfectly fitted.

## Limitations
Single city/station; ERA5 grid vs airport station mismatch; live inputs come from a different data product; threshold is a project choice, not an official warning category; probabilities are model estimates, not guaranteed risks. Dangerous-label prevalence shifted from 24.2% in validation to 47.6% in the test period, so test metrics should not be assumed to predict future performance. **Not an official warning system** – follow Pakistan Meteorological Department advisories.
