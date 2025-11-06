"""
Signal processing endpoints (FFT, etc.).
"""

from fastapi import APIRouter, HTTPException, Depends
import numpy as np
from server.dto import FFTRequest, FFTResponse
from server.auth import get_auth_handler

router = APIRouter(
    prefix="/processing",
    tags=["Processing"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/fft",
    response_model=FFTResponse,
    dependencies=[Depends(get_auth_handler().verify_api_key)],
)
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
        freq = np.fft.fftfreq(n, d=1.0)

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
