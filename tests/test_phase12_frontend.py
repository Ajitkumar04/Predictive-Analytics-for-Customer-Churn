from frontend.app import APP_TITLE, app


def test_frontend_app_has_prediction_ui():
    assert app is not None
    assert APP_TITLE == "Customer Churn Prediction"
    assert "Churn" in APP_TITLE
