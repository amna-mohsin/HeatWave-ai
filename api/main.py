"""FastAPI backend + static HTML frontend.   Run:  uvicorn api.main:app --reload"""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .schemas import PredictResponse, WeatherInput
from .services import ModelNotAvailable, load_model, predict_live, predict_scenario

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
app = FastAPI(title="HeatWave AI - Karachi", version="1.0",
              description="Week-ahead dangerous-heat warning (heat index >= 40.6 C on any of days t+5..t+7).")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/health")
def health():
    try:
        load_model(); return {"status": "ok", "model_loaded": True}
    except ModelNotAvailable as e:
        return {"status": "degraded", "model_loaded": False, "detail": str(e)}


@app.get("/model-info")
def model_info():
    try:
        _, meta = load_model()
    except ModelNotAvailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    keep = ["project", "model", "target", "target_definition", "prediction_contract", "decision_threshold",
            "primary_metric", "feature_version", "trained_utc", "data_downloaded_utc", "dataset_rows",
            "split", "test_metrics_tuned_threshold", "features"]
    return {k: meta.get(k) for k in keep}


@app.post("/predict", response_model=PredictResponse)
def predict(body: WeatherInput):
    try:
        data = body.model_dump(exclude={"date"})
        return predict_scenario(data, date=body.date)
    except ModelNotAvailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:                                   # never leak a stack trace to the client
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(e).__name__}")


@app.get("/predict/live", response_model=PredictResponse)
def live():
    try:
        return predict_live()
    except ModelNotAvailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=502, detail="Could not fetch live weather from Open-Meteo.")
