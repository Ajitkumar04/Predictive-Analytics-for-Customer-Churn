from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import CustomerInput, PredictionResponse
from src.utils.config import PROJECT_ROOT

router = APIRouter()


MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


@router.get("/health")
def health_check() -> dict[str, object]:
    """Return service status and whether the model artifact is available."""
    model_loaded = MODEL_PATH.exists()
    return {"status": "healthy", "model_loaded": model_loaded}


@router.post("/predict", response_model=PredictionResponse)
def predict_customer(payload: CustomerInput) -> PredictionResponse:
    """Return a churn prediction using the saved model artifact if available."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact is not available yet. Train and save the model first.")

    model = joblib.load(MODEL_PATH)

    row = pd.DataFrame(
        [
            {
                "subscription_type": payload.subscription_type,
                "payment_method": payload.payment_method,
                "paperless_billing": payload.paperless_billing,
                "content_type": payload.content_type,
                "multi_device_access": payload.multi_device_access,
                "device_registered": payload.device_registered,
                "genre_preference": payload.genre_preference,
                "gender": payload.gender,
                "parental_control": payload.parental_control,
                "subtitles_enabled": payload.subtitles_enabled,
                "account_age": payload.account_age,
                "monthly_charges": payload.monthly_charges,
                "total_charges": payload.total_charges,
                "viewing_hours_per_week": payload.viewing_hours_per_week,
                "support_tickets_per_month": payload.support_tickets_per_month,
                "average_viewing_duration": payload.average_viewing_duration,
                "content_downloads_per_month": payload.content_downloads_per_month,
                "user_rating": payload.user_rating,
                "watchlist_size": payload.watchlist_size,
            }
        ]
    )

    probability = float(model.predict_proba(row)[0, 1]) if hasattr(model, "predict_proba") else 0.5
    prediction = int(probability >= 0.5)
    label = "Churn" if prediction == 1 else "No Churn"
    risk_level = "High" if probability >= 0.7 else "Medium" if probability >= 0.4 else "Low"

    return PredictionResponse(
        prediction=prediction,
        prediction_label=label,
        probability=probability,
        risk_level=risk_level,
    )
