"""
API routes for the Rotary Insight Inference Server.
"""

from server.routes.models import router as models_router
from server.routes.predict import router as predict_router
from server.routes.health import router as health_router
from server.routes.processing import router as processing_router
from server.routes.examples import router as examples_router

__all__ = [
    "models_router",
    "predict_router",
    "health_router",
    "processing_router",
    "examples_router",
]
