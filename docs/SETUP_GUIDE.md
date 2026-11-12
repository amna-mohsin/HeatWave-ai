# HeatWave AI - Setup & "how does it work" guide (easy words)

## A. What you need first
1. **Python 3.12** from https://www.python.org/downloads/ (the committed model and exact dependency pins were tested with Python 3.12.3; on Windows, tick "Add Python to PATH").
2. **Git** from https://git-scm.com/downloads and a free **GitHub** account (https://github.com).
3. Internet (only for downloading the data once, and for the live weather in the apps).
4. A code editor: VS Code (https://code.visualstudio.com/).

## B. Data sources and licences (put these in your report)
| | Open-Meteo (ERA5) | NOAA GSOD |
|---|---|---|
| Used for | input features (daily weather) | target labels (station heat index) |
| Page | https://open-meteo.com/en/docs/historical-weather-api | https://www.ncei.noaa.gov/products/land-based-station/global-summary-of-the-day |
| Download | API (our script calls it) | https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/ |
| Licence | CC BY 4.0 - https://creativecommons.org/licenses/by/4.0/ ; free API is for non-commercial use | NOAA open data; cite NOAA NCEI. Some international commercial uses may be subject to WMO Resolution 40 |
| Credit line | "Weather data by Open-Meteo.com" | "NOAA NCEI. 1999. Global Surface Summary of the Day - GSOD. 1.0. Accessed <date>" |
| Station | grid cell at 24.86, 67.01 | Karachi Jinnah Intl Airport, id 41780099999 |
Extra: Open-Meteo terms https://open-meteo.com/en/terms - GSOD documentation https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt - heat index formula https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
The **download date** is written for you into `data/DATA_SOURCES.md` when you run step 3 below. Never type a guessed date.

## C. Step-by-step (Windows commands first, Mac/Linux in brackets)
Open a terminal inside the project folder (VS Code: Terminal > New Terminal).
```
py -3.12 -m venv .venv                (python3.12 -m venv .venv)
.venv\Scripts\activate                (source .venv/bin/activate)
python -m pip install -r requirements.txt
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.
You know it worked when the prompt starts with `(.venv)`.

`requirements.txt` is the exact dependency lock used by the included trained model. `requirements.in` is a separate set of minimum-version constraints used only when deliberately refreshing dependencies. For a refreshed environment, retrain and test the model, then regenerate `requirements.txt` with `python scripts/pin_requirements.py` before deployment.

Then run the 4 pipeline steps, one by one, and read the output:
```
python -m src.collect       # 1. downloads both datasets (a few minutes)
python -m src.data_loader   # 2. cleans, merges, builds features + labels. Prints rows/features/class balance
python -m src.eda           # 3. makes 10 charts in reports/figures
python -m src.train         # 4. tunes 6 classifiers vs Dummy baseline and saves the best
python -m pytest -q         # 5. checks everything
python scripts/build_report.py  # creates a four-page report draft from saved metrics and figures
```
Then the apps:
```
streamlit run app/Home.py           # dashboard opens in your browser
uvicorn api.main:app --reload       # open http://127.0.0.1:8000
```
Keep each app running in its own terminal. To open a second terminal in VS Code, choose **Terminal > New Terminal**, then activate `.venv` again before running the second command.
Always run commands from the project root folder (the one containing README.md).
The dashboard's charts are view-only; zooming, panning, hover tooltips and chart-edit controls are disabled.
Short, plain-language explanations for the EDA and model charts are in `reports/GRAPH_INTERPRETATIONS.md` and appear under the relevant dashboard charts.

## D. How it works (the simple version)
1. **Data.** The current collection has about 45 years of daily Karachi weather from two places. One (Open-Meteo/ERA5) gives temperature, humidity, wind, pressure etc.
2. **Label.** The second (airport station) tells us how hot it *actually felt*. We turn its temperature and dew point into the "heat index". If the heat index is 40.6 C or more on any of the days 5, 6 or 7 after a given day, that given day gets label 1 (danger), otherwise 0.
3. **Learning.** The model sees "today's weather + the last week" and the label, thousands of times, and learns which patterns come before dangerous heat.
4. **Honest testing.** We split by time: oldest 70% to learn, next 15% to choose the best model, newest 15% to test after model selection. Never random, because that would let the model peek at the future. The training/validation graph checks for a generalization gap; a model cannot be guaranteed perfectly fitted.
5. **Saving.** The best model (with all preprocessing inside) is saved as one file, `models/heatwave_pipeline.joblib`.
6. **Using it.** The apps load that file. You type today's weather (or click "live") -> the app builds the same features -> the model outputs a probability -> if it is above the chosen threshold you get a danger warning.
Both apps (Streamlit and FastAPI+HTML) use the same `src/predict.py`, so they always agree.

## E. What to open during your demo
Streamlit Home -> Heat Prediction (click Predict) -> Model Insights (table + charts) -> HF Space (same app) -> FastAPI page on localhost.

## F. If something breaks
| Problem | Fix |
|---|---|
| `No module named src` | You are not in the project root folder |
| `Model not found` | run `python -m src.train` first |
| collect: a year is "not available" | NOAA may not have published it yet; set END_DATE in `src/config.py` to the last available year |
| Open-Meteo 429 error | free limit hit; wait a few minutes and re-run (script retries by itself) |
| Streamlit Cloud can't load the model | `requirements.txt` not pinned, or `models/` not committed to GitHub |
| Live prediction fails in the app | no internet / Open-Meteo down; use "Manual scenario" |

## G. CT-378 items to finish before submission
The local project includes the data pipeline, six tuned classifiers plus baseline, saved model, EDA/evaluation charts, tests, Streamlit dashboard, and local FastAPI + HTML app. You still need to publish and test the public Streamlit Cloud and Hugging Face deployments, add your actual deployment links and screenshots, fill in all group members' real names/contributions, and prepare the 3–4 page PDF report. These steps require your accounts and team information.
