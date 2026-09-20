from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

APP_TITLE = "Customer Churn Prediction"
app = st


def _risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def _predict_locally(payload: dict[str, Any]) -> dict[str, Any]:
    """Provide a lightweight fallback prediction when the API is unavailable."""
    score = 0.0
    score += 0.15 if payload["support_tickets_per_month"] > 2 else 0.0
    score += 0.12 if payload["monthly_charges"] > 40 else 0.0
    score += 0.10 if payload["user_rating"] <= 2 else 0.0
    score += 0.08 if payload["watchlist_size"] > 10 else 0.0
    score += 0.12 if payload["account_age"] < 12 else 0.0
    score += 0.10 if payload["payment_method"] in {"Bank Transfer", "Credit Card"} else 0.0
    score += 0.10 if payload["subscription_type"] == "Basic" else 0.0

    probability = min(max(score, 0.05), 0.95)
    prediction = 1 if probability >= 0.5 else 0
    return {
        "prediction": prediction,
        "prediction_label": "Churn" if prediction == 1 else "No Churn",
        "probability": round(probability, 4),
        "risk_level": _risk_level(probability),
    }


def _predict_via_api(payload: dict[str, Any]) -> dict[str, Any] | None:
    api_url = os.getenv("API_URL", "http://localhost:8000/predict")
    try:
        response = requests.post(api_url, json=payload, timeout=5)
        if response.ok:
            return response.json()
    except requests.RequestException:
        return None
    return None


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Enter customer details to estimate churn risk.")

    with st.form("customer_form"):
        col1, col2 = st.columns(2)

        with col1:
            subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            payment_method = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check", "Debit Card"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            content_type = st.selectbox("Content Type", ["Movies", "TV Shows", "Both"])
            multi_device_access = st.selectbox("Multi-device Access", ["Yes", "No"])
            device_registered = st.selectbox("Device Registered", ["Smart TV", "Phone", "Tablet", "Laptop"])
            genre_preference = st.selectbox("Genre Preference", ["Drama", "Comedy", "Action", "Documentary"])
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            parental_control = st.selectbox("Parental Control", ["Yes", "No"])
            subtitles_enabled = st.selectbox("Subtitles Enabled", ["Yes", "No"])

        with col2:
            account_age = st.number_input("Account Age", min_value=0, max_value=100, value=12)
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=15.5, step=0.5)
            total_charges = st.number_input("Total Charges", min_value=0.0, max_value=2000.0, value=186.0, step=1.0)
            viewing_hours_per_week = st.number_input("Viewing Hours per Week", min_value=0.0, max_value=100.0, value=12.0, step=1.0)
            support_tickets_per_month = st.number_input("Support Tickets per Month", min_value=0, max_value=50, value=2)
            average_viewing_duration = st.number_input("Average Viewing Duration", min_value=0.0, max_value=500.0, value=45.0, step=1.0)
            content_downloads_per_month = st.number_input("Content Downloads per Month", min_value=0, max_value=100, value=8)
            user_rating = st.slider("User Rating", 1, 5, 3)
            watchlist_size = st.number_input("Watchlist Size", min_value=0, max_value=100, value=20)

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        payload = {
            "subscription_type": subscription_type,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "content_type": content_type,
            "multi_device_access": multi_device_access,
            "device_registered": device_registered,
            "genre_preference": genre_preference,
            "gender": gender,
            "parental_control": parental_control,
            "subtitles_enabled": subtitles_enabled,
            "account_age": int(account_age),
            "monthly_charges": float(monthly_charges),
            "total_charges": float(total_charges),
            "viewing_hours_per_week": float(viewing_hours_per_week),
            "support_tickets_per_month": int(support_tickets_per_month),
            "average_viewing_duration": float(average_viewing_duration),
            "content_downloads_per_month": int(content_downloads_per_month),
            "user_rating": int(user_rating),
            "watchlist_size": int(watchlist_size),
        }

        result = _predict_via_api(payload)
        if result is None:
            result = _predict_locally(payload)

        st.subheader("Prediction Result")
        st.metric("Churn Probability", f"{result['probability'] * 100:.2f}%")
        st.metric("Risk Level", result["risk_level"])
        st.write(f"Prediction: {result['prediction_label']}")

        if result["prediction"] == 1:
            st.warning("The model indicates a higher churn risk for this customer.")
        else:
            st.success("The model indicates a lower churn risk for this customer.")


if __name__ == "__main__":
    main()

