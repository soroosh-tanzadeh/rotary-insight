"""
Pydantic model for error responses.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

