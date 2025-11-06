"""
Pydantic models for FFT transformation requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class FFTRequest(BaseModel):
    """Request model for FFT transformation"""

    signal: List[float] = Field(..., description="Time-domain signal data")
    n: Optional[int] = Field(
        None,
        description="FFT length (number of points). If not provided, uses len(signal).",
    )


class FFTResponse(BaseModel):
    """Response model for FFT output"""

    n: int = Field(..., description="FFT length used for transformation")
    frequencies: List[float] = Field(..., description="Frequencies (normalized units)")
    magnitudes: List[float] = Field(..., description="Magnitude spectrum (L2 norm)")
