"""
FastAPI inference server for bearing fault classification models.

This server provides authenticated REST API endpoints for:
- Model inference
- Model listing
- Health checks

Authentication is done via API keys in the X-API-Key header.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import io
import time
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .models import (
    InferenceRequest,
    InferenceResponse,
    PredictionResult,
    ModelsListResponse,
    HealthResponse,
    ErrorResponse,
    FFTRequest,
    FFTResponse,
    ExamplesListResponse,
    ExampleFile,
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


@app.post(
    "/predict-csv",
    response_model=InferenceResponse,
    summary="Perform inference on uploaded CSV file",
    dependencies=[Depends(get_auth_handler().verify_api_key)],
)
async def predict_from_csv(
    file: UploadFile = File(..., description="CSV file containing signal data (one channel)"),
    model_name: str = Form(..., description="Model name to use for inference"),
    return_probabilities: bool = Form(False, description="Return class probabilities"),
):
    """
    Perform model inference on a CSV file containing one or more signal samples.

    Expected CSV format:
    ```
    row,ch1
    0.0,-0.06724814371257484
    1.0,-0.22822135728542914
    ...
    ```

    Example curl:
    ```
    curl -X POST "http://127.0.0.1:3000/csv/predict" \
        -H "X-API-Key: your_api_key" \
        -F "file=@sample1_Normal.csv" \
        -F "model_name=transformer_encoder_cwru_512"
    ```
    """
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model manager is not initialized.",
        )

    start_time = time.time()

    try:
        # Read CSV into pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Validate column
        if "ch1" not in df.columns:
            raise ValueError("CSV file must contain a 'ch1' column.")

        # Extract signal
        signal = df["ch1"].values.astype(np.float32)

        # Get expected window size from model config
        config = model_manager.get_model_config(model_name)
        window_size = config.get("window_size", len(signal))

        # Pad or trim signal to expected length
        if len(signal) < window_size:
            pad_width = window_size - len(signal)
            signal = np.pad(signal, (0, pad_width), mode="constant")
        elif len(signal) > window_size:
            signal = signal[:window_size]

        # Reshape to (batch=1, channels=1, signal_length)
        input_data = np.expand_dims(np.expand_dims(signal, axis=0), axis=0)

        # Perform inference
        predictions, probabilities = model_manager.predict(
            model_name=model_name,
            data=input_data,
            return_probabilities=return_probabilities,
        )

        # Get class names
        class_names = model_manager.get_class_names(model_name)

        # Build prediction result
        pred_class = int(predictions[0])
        result = PredictionResult(
            predicted_class=pred_class,
            class_name=class_names[pred_class],
            confidence=float(probabilities[0, pred_class]) if probabilities is not None else 0.0,
            probabilities=probabilities[0].tolist() if probabilities is not None else None,
        )

        processing_time = (time.time() - start_time) * 1000

        return InferenceResponse(
            model_name=model_name,
            predictions=[result],
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@app.post("/fft", response_model=FFTResponse,dependencies=[Depends(get_auth_handler().verify_api_key)],)
async def perform_fft(request: FFTRequest):
    """
    Perform Fast Fourier Transform (FFT) on a given time-domain signal.

    Example:
    ```json
    {
        "signal": [0.1, 0.2, 0.3, 0.4],
        "n": 512
    }
    ```
    """
    try:
        signal = np.array(request.signal, dtype=np.float32)
        n = request.n or len(signal)

        # Pad or truncate to n length
        if len(signal) < n:
            signal = np.pad(signal, (0, n - len(signal)), mode="constant")
        elif len(signal) > n:
            signal = signal[:n]

        # Perform FFT
        fft_values = np.fft.fft(signal, n=n)

        # Compute magnitude (L2 norm)
        magnitude = np.abs(fft_values)

        # Frequencies (normalized 0 → Nyquist)
        freq = np.fft.fftfreq(n, d=1.0)  # d=1 means normalized frequency spacing

        # Only keep positive half (for real signals)
        half = n // 2
        freq = freq[:half]
        magnitude = magnitude[:half]

        return FFTResponse(
            n=n,
            frequencies=freq.tolist(),
            magnitudes=magnitude.tolist(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"FFT computation failed: {str(e)}",
        )

# Directory containing CSV examples
EXAMPLES_DIR = os.getenv("EXAMPLES_DIR", "samples")

@app.get(
    "/examples",
    response_model=ExamplesListResponse,
    tags=["Examples"],
    summary="List example CSV files for fault detection",
)
async def list_example_files():
    """
    List all available example CSV files for fault detection.
    The files are expected to follow the naming pattern:
    `sample{index}_{fault_name}.csv`
    """
    if not os.path.exists(EXAMPLES_DIR):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Examples directory not found: {EXAMPLES_DIR}",
        )

    files = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".csv")]
    examples = []

    pattern = re.compile(r"sample(\d+)_(.+)\.csv", re.IGNORECASE)

    for f in files:
        match = pattern.match(f)
        if match:
            index = int(match.group(1))
            fault_name = match.group(2)
            examples.append(
                ExampleFile(
                    filename=f,
                    sample_index=index,
                    fault_name=fault_name,
                    download_url=f"/examples/download/{f}",
                )
            )

    return ExamplesListResponse(examples=examples, total_count=len(examples))


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
