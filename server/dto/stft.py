"""
Pydantic models for STFT transformation requests and responses.
"""

from pydantic import BaseModel
from typing import List, Optional

class STFTRequest(BaseModel):
    signal: List[float]
    n_fft: int = 512
    hop_length: Optional[int] = None
    win_length: Optional[int] = None

class STFTResponse(BaseModel):
    stft: List[List[float]]  # Magnitude spectrogram
    frequencies: List[float]
    times: List[float]
