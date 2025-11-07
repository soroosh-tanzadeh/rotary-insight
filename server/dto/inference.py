"""
Pydantic models for inference requests and responses.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


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

