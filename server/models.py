"""
Pydantic models for request and response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import numpy as np


class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""

    data: List[List[List[float]]] = Field(
        ...,
        description="Input signal data with shape (batch_size, channels, signal_length)",
        example=[[[0.1, 0.2, 0.3, 0.4] * 128]],  # Example for window_size=512
    )

    model_name: str = Field(
        ...,
        description="Name of the model to use for inference",
        example="transformer_encoder_cwru_512",
    )

    return_probabilities: bool = Field(
        default=False,
        description="Whether to return class probabilities instead of just predictions",
    )

    @field_validator("data")
    @classmethod
    def validate_data_shape(cls, v):
        """Validate that data has the correct shape."""
        if not v:
            raise ValueError("Data cannot be empty")

        # Check that all samples have the same shape
        if not all(len(sample) == len(v[0]) for sample in v):
            raise ValueError("All samples must have the same number of channels")

        if not all(len(channel) == len(v[0][0]) for sample in v for channel in sample):
            raise ValueError("All channels must have the same length")

        return v


class PredictionResult(BaseModel):
    """Single prediction result."""

    predicted_class: int = Field(..., description="Predicted class index")
    class_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., description="Confidence score (max probability)")
    probabilities: Optional[List[float]] = Field(
        None, description="Class probabilities (if return_probabilities=True)"
    )


class InferenceResponse(BaseModel):
    """Response model for inference endpoint."""

    model_name: str = Field(..., description="Name of the model used")
    predictions: List[PredictionResult] = Field(..., description="Prediction results")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type (e.g., 'pytorch')")
    window_size: int = Field(..., description="Expected input window size")
    dataset_name: str = Field(..., description="Dataset the model was trained on")
    task: str = Field(..., description="Task type (e.g., 'classification')")
    num_classes: int = Field(..., description="Number of output classes")
    class_names: List[str] = Field(..., description="Names of the classes")
    loaded: bool = Field(..., description="Whether the model is currently loaded")


class ModelsListResponse(BaseModel):
    """Response model for listing available models."""

    models: Dict[str, ModelInfo] = Field(..., description="Available models")
    total_count: int = Field(..., description="Total number of models")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status")
    mlflow_uri: str = Field(..., description="MLflow tracking URI")
    models_loaded: int = Field(..., description="Number of models loaded")
    available_models: List[str] = Field(
        ..., description="List of available model names"
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class FFTRequest(BaseModel):
    """Request model for FFT transformation"""
    signal: List[float] = Field(..., description="Time-domain signal data")
    n: Optional[int] = Field(
        None,
        description="FFT length (number of points). If not provided, uses len(signal)."
    )


class FFTResponse(BaseModel):
    """Response model for FFT output"""
    n: int = Field(..., description="FFT length used for transformation")
    frequencies: List[float] = Field(..., description="Frequencies (normalized units)")
    magnitudes: List[float] = Field(..., description="Magnitude spectrum (L2 norm)")
