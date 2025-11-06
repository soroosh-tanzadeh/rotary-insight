"""
Pydantic model for health check response.
"""

from pydantic import BaseModel, Field
from typing import List


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status")
    mlflow_uri: str = Field(..., description="MLflow tracking URI")
    models_loaded: int = Field(..., description="Number of models loaded")
    available_models: List[str] = Field(
        ..., description="List of available model names"
    )
