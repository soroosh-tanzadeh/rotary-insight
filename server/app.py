"""
FastAPI inference server for bearing fault classification models.

This server provides authenticated REST API endpoints for:
- Model inference
- Model listing
- Health checks

Authentication is done via API keys in the X-API-Key header.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
from server.dto import ErrorResponse
from server.auth import get_auth_handler
from server.inference import ModelManager, setup_model_manager
from server.routes import (
    models_router,
    predict_router,
    health_router,
    processing_router,
    examples_router,
)

# Configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "model_serve_config.json")
HOST = os.getenv("ROTARY_INSIGHT_HOST", "0.0.0.0")
PORT = int(os.getenv("ROTARY_INSIGHT_PORT", "8000"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Replaces deprecated @app.on_event("startup") and @app.on_event("shutdown").
    """
    try:
        get_auth_handler()
        print("✓ Authentication initialized")
    except Exception as e:
        print(f"✗ Failed to initialize authentication: {e}")
        raise

    # Initialize model manager
    try:
        setup_model_manager(MODEL_CONFIG_PATH, MLFLOW_TRACKING_URI)
        print("✓ Model manager initialized")

    except Exception as e:
        print(f"✗ Failed to initialize model manager: {e}")
        raise

    print("=" * 50)
    print(f"Server ready at http://{HOST}:{PORT}")
    print(f"API Documentation: http://{HOST}:{PORT}/docs")
    print("=" * 50)

    yield

    # Shutdown
    print("\nShutting down Rotary Insight Inference Server...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Rotary Insight API",
    description="Bearing Fault Classification Inference API using Transformer models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(health_router)
app.include_router(models_router)
app.include_router(predict_router)
app.include_router(processing_router)
app.include_router(examples_router)


# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail, detail=getattr(exc, "detail", None)
        ).model_dump(),
    )
