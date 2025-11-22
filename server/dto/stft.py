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
    file_name: str
    file_type: str = "image/png"
    image_base64: str