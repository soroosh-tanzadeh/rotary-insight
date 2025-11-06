"""
Pydantic models for model information and listing.
"""

from pydantic import BaseModel, Field
from typing import List, Dict


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

