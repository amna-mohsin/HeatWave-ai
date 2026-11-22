<div align="center">

# HeatWave AI

### Week-ahead dangerous-heat risk for Karachi, Pakistan

An educational machine-learning project that estimates whether the Karachi airport station heat index will reach **40.6 °C (105 °F)** on any of the next **5–7 days**.

[Overview](#about) · [Features](#features) · [Setup](#getting-started) · [Models](#machine-learning) · [Authors](#authors) · [License](#license)

</div>

## About

HeatWave AI is a **binary classification** project. It combines daily weather features with a historical station-based target label to estimate a probability of dangerous heat. It is intended for learning and demonstration, not as an official warning service.

The project compares four tuned classifiers against two baselines: a Dummy that ignores the weather, and a persistence rule ("hot today, danger next week") that uses only today's heat index. Model selection uses validation data; the newest period is held out for final testing. See the [model card](docs/model_card.md) and [graph interpretations](reports/GRAPH_INTERPRETATIONS.md) for methodology and plain-language explanations.

## Deployments

| Platform | Link |
|---|---|
| Streamlit Community Cloud | [https://heatwave-ai.streamlit.app/](https://heatwave-ai.streamlit.app/) |
| Hugging Face (model + Space) | _Add the Hub model and Space URLs here after publishing (see [`deploy/huggingface/`](deploy/huggingface/))_ |
| Local FastAPI web app | [http://127.0.0.1:8000](http://127.0.0.1:8000) after `uvicorn api.main:app` (`POST /predict`, docs at `/docs`) |

The project report is in [`reports/HeatWave_AI_Report.pdf`](reports/HeatWave_AI_Report.pdf).

## Features

- **Reproducible data workflow:** collect weather data, clean and join it, build features and labels, then produce EDA charts.
- **Four classifiers plus two baselines:** Logistic Regression, Random Forest, Gradient Boosting and SVM, compared with a Dummy baseline (beats guessing?) and a persistence baseline (beats "hot stays hot"?).
- **Chronological evaluation:** time-ordered train, validation, and test splits help avoid training on future observations.
- **Shared prediction code:** the Streamlit dashboard and FastAPI app use the same saved pipeline and prediction logic.
- **Readable results:** model comparisons, evaluation graphs, and short interpretations are included.
- **Data attribution:** collection dates and source credits are recorded in [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md).

## Tech stack

Python · pandas · scikit-learn · Streamlit · FastAPI · Plotly · pytest

## How it works

```text
Open-Meteo ERA5 weather ──┐
                         ├─> cleaned features ─> trained classifier ─> heat-risk probability
NOAA GSOD station data ───┘       issue day t                         days t+5 to t+7
```

Open-Meteo provides weather input features; NOAA GSOD provides station observations used to calculate the target label. A day is labelled dangerous when the station heat index reaches 40.6 °C or more on any day from five to seven days later. The model uses issue-day weather and its history, not future weather values.

## Project structure

```text
app/                 Streamlit dashboard
api/                 FastAPI endpoints
frontend/            HTML/CSS/JavaScript served by the API
src/                 Data collection, feature engineering, training, prediction
data/                Source credits and raw/processed datasets
models/              Trained pipeline and model metadata
reports/             Evaluation figures, metrics, and graph explanations
notebooks/            Step-by-step project notebooks
tests/                Automated tests
docs/                 Setup guide and model card
```

## Getting started

### Requirements

- Python 3.12 (the included model and pinned dependencies were tested with Python 3.12.3)
- Git, if cloning the project
- Internet access for the data-collection step and live weather predictions

### Install

Open a terminal in the project root (the folder containing this README), then run:

**Windows**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins the versions used by the included trained model. `requirements.in` is for intentionally refreshing dependencies; after changing dependencies, retrain and test the model before updating the pins.

### Run the applications

Start each app in its own activated terminal:

```bash
source .venv/bin/activate
streamlit run app/Home.py
```

```bash
uvicorn api.main:app --reload
```

The Streamlit dashboard opens in your browser. The FastAPI page is at [http://127.0.0.1:8000](http://127.0.0.1:8000), with interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Rebuild the dataset and model

The project includes data and a trained model so you can run the apps without recollecting data. To reproduce the pipeline, run these from the project root:

```bash
python -m src.collect
python -m src.data_loader
python -m src.eda
python -m src.train
python -m pytest -q
```

Collection downloads data from the sources described in [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md). Training runs five-fold time-series tuning for the four classifiers and compares them with the Dummy and persistence baselines; a full run may take 10–30 minutes. To run the smaller development smoke test instead, use `HEATWAVE_FAST=1 python -m src.train` (do not use that mode for final results).

## Machine learning

The task is binary classification: predict `danger_5to7d` from weather features for issue day *t* and its past. The target is 1 if station heat index is at least 40.6 °C on any of days *t+5* through *t+7*.

The current saved pipeline selected Logistic Regression using validation PR-AUC. It was evaluated on a chronological held-out test set; the latest recorded PR-AUC is **0.962**, ROC-AUC **0.966**, and F1 **0.889** at the validation-selected threshold (about **0.378**). These are historical test results, not a promise of future performance. The dangerous-label rate differs substantially between validation (24.2%) and test (47.6%), so temporal distribution shift is an important limitation. The training-to-validation PR-AUC gap is about 0.100 and should be treated as a reason for caution, not hidden or described as a perfect fit.

See [`reports/GRAPH_INTERPRETATIONS.md`](reports/GRAPH_INTERPRETATIONS.md) for concise explanations of the evaluation and EDA charts. The threshold is a project setting, not an official public-safety threshold.

## Data sources and attribution

| Source | Project role | Terms and credit |
|---|---|---|
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | ERA5 weather input features | CC BY 4.0; credit “Weather data by Open-Meteo.com”. The free API is for non-commercial use; see [Open-Meteo terms](https://open-meteo.com/en/terms). |
| [NOAA NCEI GSOD](https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/) | Karachi airport station observations for target labels | NOAA open data; cite NOAA NCEI. See [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md) for the full credit and caveats. |

The project’s [MIT License](LICENSE) applies to project-authored software. It does not replace the source-specific terms or attribution requirements for datasets and other third-party materials.

## Tests

Run the test suite from the project root:

```bash
python -m pytest -q
```

## Limitations

- This is a single-city study; ERA5 grid data and airport station measurements describe different products and locations.
- A live prediction uses current/recent weather inputs and does not guarantee what will happen in the future.
- The target definition and alert threshold are project choices, not official heat warnings.
- Model performance can change as weather and dangerous-day prevalence shift over time.
- For safety decisions, follow official Pakistan Meteorological Department advisories.

## Authors

- #### Hamna Ali Khan — [GitHub](https://github.com/HamnaAliKhan)
- #### Amna Mohsin  — [GitHub](https://github.com/amna-mohsin) · [LinkedIn](https://www.linkedin.com/in/amna-m98/)

## License

Project-authored software is licensed under the [MIT License](LICENSE). Dataset and third-party material terms remain as described above.

## Academic use

This project was developed for educational use. If you reuse or adapt it, credit the authors and follow your institution’s academic-integrity requirements.
