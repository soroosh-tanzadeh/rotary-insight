"""
FastAPI inference server for bearing fault classification models.

This server provides authenticated REST API endpoints for:
- Model inference
- Model listing
- Health checks

Authentication is done via API keys in the X-API-Key header.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import time
import numpy as np
from dotenv import load_dotenv

from .models import (
    InferenceRequest,
    InferenceResponse,
    PredictionResult,
    ModelsListResponse,
    HealthResponse,
    ErrorResponse,
)
from .auth import get_auth_handler
from .inference import ModelManager

# Load environment variables
load_dotenv()

# Configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "model_serve_config.json")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Global model manager
model_manager: ModelManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Replaces deprecated @app.on_event("startup") and @app.on_event("shutdown").
    """
    # Startup
    global model_manager

    print("=" * 50)
    print("Starting Rotary Insight Inference Server")
    print("=" * 50)

    # Initialize auth handler
    try:
        get_auth_handler()
        print("✓ Authentication initialized")
    except Exception as e:
        print(f"✗ Failed to initialize authentication: {e}")
        raise

    # Initialize model manager
    try:
        model_manager = ModelManager(
            config_path=MODEL_CONFIG_PATH, mlflow_uri=MLFLOW_TRACKING_URI
        )
        print("✓ Model manager initialized")

        # Optionally load all models on startup (comment out if you want lazy loading)
        # model_manager.load_all_models()

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


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Rotary Insight Inference API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
)
async def health_check():
    """
    Check the health status of the API.

    Returns service status and information about loaded models.
    """
    models_info = model_manager.list_models()

    return HealthResponse(
        status="healthy",
        mlflow_uri=MLFLOW_TRACKING_URI,
        models_loaded=sum(1 for m in models_info.values() if m["loaded"]),
        available_models=list(models_info.keys()),
    )


@app.get(
    "/models",
    response_model=ModelsListResponse,
    tags=["Models"],
    summary="List available models",
    dependencies=[Depends(get_auth_handler().verify_api_key)],
)
async def list_models():
    """
    List all available models with their configurations.

    Returns detailed information about each model including:
    - Model type and window size
    - Dataset and task information
    - Class names
    - Loading status
    """
    try:
        models_info = model_manager.list_models()
        return ModelsListResponse(models=models_info, total_count=len(models_info))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


@app.post(
    "/predict",
    response_model=InferenceResponse,
    tags=["Inference"],
    summary="Perform inference",
    dependencies=[Depends(get_auth_handler().verify_api_key)],
)
async def predict(request: InferenceRequest):
    """
    Perform inference on input signal data.

    Args:
        request: Inference request containing:
            - data: Input signal data with shape (batch_size, channels, signal_length)
            - model_name: Name of the model to use
            - return_probabilities: Whether to return class probabilities

    Returns:
        Inference results including predictions, class names, and confidence scores.

    Example:
        ```json
        {
            "data": [[[0.1, 0.2, ...]]],  # shape: (1, 1, 512)
            "model_name": "transformer_encoder_cwru_512",
            "return_probabilities": true
        }
        ```
    """
    start_time = time.time()

    try:
        # Convert data to numpy array
        data = np.array(request.data, dtype=np.float32)

        # Perform inference
        predictions, probabilities = model_manager.predict(
            model_name=request.model_name,
            data=data,
            return_probabilities=request.return_probabilities,
        )

        # Get class names
        class_names = model_manager.get_class_names(request.model_name)

        # Build response
        results = []
        for i, pred_class in enumerate(predictions):
            result = PredictionResult(
                predicted_class=int(pred_class),
                class_name=class_names[pred_class],
                confidence=(
                    float(probabilities[i, pred_class])
                    if probabilities is not None
                    else 0.0
                ),
                probabilities=(
                    probabilities[i].tolist() if probabilities is not None else None
                ),
            )
            results.append(result)

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return InferenceResponse(
            model_name=request.model_name,
            predictions=results,
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@app.post(
    "/models/{model_name}/load",
    response_model=dict,
    tags=["Models"],
    summary="Load a specific model",
    dependencies=[Depends(get_auth_handler().verify_api_key)],
)
async def load_model(model_name: str):
    """
    Explicitly load a model into memory.

    Models are loaded lazily on first inference by default.
    Use this endpoint to pre-load models for faster first inference.
    """
    try:
        model_manager.load_model(model_name)
        return {
            "status": "success",
            "message": f"Model '{model_name}' loaded successfully",
            "model_name": model_name,
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@app.delete(
    "/models/{model_name}/unload",
    response_model=dict,
    tags=["Models"],
    summary="Unload a specific model",
    dependencies=[Depends(get_auth_handler().verify_api_key)],
)
async def unload_model(model_name: str):
    """
    Unload a model from memory.

    Use this to free up memory when a model is no longer needed.
    """
    try:
        model_manager.unload_model(model_name)
        return {
            "status": "success",
            "message": f"Model '{model_name}' unloaded successfully",
            "model_name": model_name,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unload model: {str(e)}",
        )


# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail, detail=getattr(exc, "detail", None)
        ).dict(),
    )


def main():
    """Run the server."""
    uvicorn.run(
        "server.main:app",
        host=HOST,
        port=PORT,
        reload=False,  # Set to True for development
        log_level="info",
    )


if __name__ == "__main__":
    main()
