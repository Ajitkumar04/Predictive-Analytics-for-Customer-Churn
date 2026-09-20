from __future__ import annotations

from fastapi import FastAPI

from api.routes.prediction import router

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")
app.include_router(router)


@app.get("/health")
def health_check() -> dict[str, object]:
    """Return the API health status and whether a model artifact is available."""
    return {"status": "healthy", "model_loaded": False}
