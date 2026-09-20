from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint_returns_status():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "model_loaded" in payload


def test_predict_endpoint_accepts_valid_payload():
    payload = {
        "subscription_type": "Premium",
        "payment_method": "Credit Card",
        "paperless_billing": "Yes",
        "content_type": "Movies",
        "multi_device_access": "Yes",
        "device_registered": "Smart TV",
        "genre_preference": "Drama",
        "gender": "Male",
        "parental_control": "No",
        "subtitles_enabled": "Yes",
        "account_age": 12,
        "monthly_charges": 15.5,
        "total_charges": 186.0,
        "viewing_hours_per_week": 12,
        "support_tickets_per_month": 2,
        "average_viewing_duration": 45,
        "content_downloads_per_month": 8,
        "user_rating": 3,
        "watchlist_size": 20,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        body = response.json()
        assert "prediction" in body
        assert "probability" in body
        assert "risk_level" in body
