from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Schema placeholder for future churn prediction request payload."""

    subscription_type: str = Field(..., min_length=1)
    payment_method: str = Field(..., min_length=1)
    paperless_billing: str = Field(..., min_length=1)
    content_type: str = Field(..., min_length=1)
    multi_device_access: str = Field(..., min_length=1)
    device_registered: str = Field(..., min_length=1)
    genre_preference: str = Field(..., min_length=1)
    gender: str = Field(..., min_length=1)
    parental_control: str = Field(..., min_length=1)
    subtitles_enabled: str = Field(..., min_length=1)
    account_age: int = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    viewing_hours_per_week: float = Field(..., ge=0)
    support_tickets_per_month: int = Field(..., ge=0)
    average_viewing_duration: float = Field(..., ge=0)
    content_downloads_per_month: int = Field(..., ge=0)
    user_rating: int = Field(..., ge=0, le=5)
    watchlist_size: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
    risk_level: str
