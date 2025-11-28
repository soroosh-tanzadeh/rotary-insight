"""
Pydantic models for STFT transformation requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class STFTRequest(BaseModel):
    signal: List[float]
    n_fft: int = Field(
        ...,
        description="Number of FFT points.",
    )
    sampling_rate: Optional[int] = Field(
        None,
        description="Sample rate of the signal (samples per second). If not provided, the frequency axis will not be normalized.",
    )
    hop_length: Optional[int] = Field(
        None,
        description="Number of samples between successive frames. If not provided, defaults to n_fft // 4.",
    )
    win_length: Optional[int] = Field(
        None,
        description="Number of samples in the FFT window. If not provided, defaults to n_fft.",
    )

class STFTResponse(BaseModel):
    file_name: str
    file_type: str = "image/png"
    image_base64: str