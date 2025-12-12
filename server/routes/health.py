"""
Health check endpoints.
"""

from fastapi import APIRouter
from server.dto import HealthResponse
from server.inference import get_model_manager

# Configuration from environment (imported in app.py)
MLFLOW_TRACKING_URI = None  # Will be set by app.py

router = APIRouter(
    tags=["Health"],
    responses={404: {"description": "Not found"}},
)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
)
async def health_check():
    """
    Check the health status of the API.

    Returns service status and information about loaded models.
    """
    models_info = get_model_manager().list_models()

    return HealthResponse(
        status="healthy",
        mlflow_uri=MLFLOW_TRACKING_URI or "unknown",
        models_loaded=sum(1 for m in models_info.values() if m["loaded"]),
        available_models=list(models_info.keys()),
    )


@router.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Rotary Insight Inference API",
        "version": "1.0.0",
        "docs": "/docs",
    }
