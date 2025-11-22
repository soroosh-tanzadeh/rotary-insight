"""
Signal processing endpoints (FFT, etc.).
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse
import numpy as np
import uuid
import torch
import os
from server.dto import FFTRequest, FFTResponse, STFTRequest, STFTResponse
from server.auth import get_auth_handler
import torchvision.transforms as transforms

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


@router.post(
    "/stft",
    response_model=STFTResponse,
    dependencies=[Depends(get_auth_handler().verify_api_key)],
)
async def perform_stft(req: STFTRequest, background_tasks: BackgroundTasks):
    signal = torch.tensor(req.signal, dtype=torch.float32).unsqueeze(0)  # [1, N]
    n_fft = req.n_fft
    hop_length = req.hop_length or n_fft // 4
    win_length = req.win_length or n_fft
    # Check that n_fft does not exceed the signal length
    if n_fft > signal.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"n_fft ({n_fft}) must be less than or equal to signal length ({signal.shape[1]})"
        )
    x = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=True,
        return_complex=True,
    )
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    spectrogram = torch.abs(x).squeeze(0) 

    spectrogram_np = spectrogram.cpu().numpy()
    
    spectrogram_db = 10 * np.log10(spectrogram_np + 1e-6) # Add small epsilon to avoid log(0)
    
    min_val = spectrogram_db.min()
    max_val = spectrogram_db.max()
    norm_spectrogram = (spectrogram_db - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else np.zeros_like(spectrogram_db)
    
    cmap = cm.get_cmap('viridis')
    colored_spectrogram_np = (cmap(norm_spectrogram)[:, :, :3] * 255).astype(np.uint8)

    spectrogram_pil = transforms.ToPILImage()(colored_spectrogram_np)
    
    spectrogram = transforms.Resize((256, 1024))(spectrogram_pil)

    temp_dir = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_name = f"spectrogram_{uuid.uuid4().hex}.png"
    file_path = os.path.join(temp_dir, file_name)
    
    spectrogram.save(file_path)
    background_tasks.add_task(os.remove, file_path)
    return FileResponse(
        path=file_path,
        filename=file_name,
        media_type="image/png",
    )
    