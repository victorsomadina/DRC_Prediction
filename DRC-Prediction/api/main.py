import os
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from api.schemas import PredictRequest, PredictResponse

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"

_model_artifact: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact on startup."""
    global _model_artifact
    if MODEL_PATH.exists():
        _model_artifact = joblib.load(MODEL_PATH)
    else:
        _model_artifact = None
    yield

app = FastAPI(
    title="TerraNova DRC Prediction API",
    description="Disaster Recovery Cost Forecasting — TerraNova Resilience Analytics",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    model_loaded = _model_artifact is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_name": _model_artifact["model_name"] if model_loaded else None,
    }


@app.post("/predict-cost", response_model=PredictResponse)
def predict_cost(request: PredictRequest):
    if _model_artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training pipeline first.",
        )

    pipeline = _model_artifact["pipeline"]

    from src.processing.feature_engineering import REGION_MAP

    region = REGION_MAP.get(request.state, "Other")
    season = (
        1 if request.declaration_year in (12, 1, 2)
        else 2 if request.declaration_year in (3, 4, 5)
        else 3 if request.declaration_year in (6, 7, 8)
        else 4
    )

    input_df = pd.DataFrame([{
        "incidentType": request.incidentType,
        "declarationType": request.declarationType,
        "region": region,
        "season": season,
        "incident_duration_days": request.incident_duration_days,
        "declaration_year": request.declaration_year,
        "rolling_5yr_freq": request.rolling_5yr_freq,
        "project_count": request.project_count,
        "avg_project_amount": request.avg_project_amount,
        "is_high_cost_type": request.is_high_cost_type,
    }])

    log_pred = pipeline.predict(input_df)[0]
    cost_pred = float(np.expm1(log_pred))

    return PredictResponse(
        predicted_recovery_cost_usd=round(cost_pred, 2),
        model_name=_model_artifact["model_name"],
        confidence_note="Prediction based on historical FEMA PA obligations. "
                        "Actual costs may vary with event severity.",
    )


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
