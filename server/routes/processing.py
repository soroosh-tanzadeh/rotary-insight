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
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

        # Frequencies
        if request.sampling_rate:
            # If sampling_rate is provided, d = 1 / sampling_rate
            d = 1.0 / request.sampling_rate
        else:
            # Normalized frequency (0 to 0.5 cycles/sample)
            d = 1.0

        freq = np.fft.fftfreq(n, d=d)

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
    signal = torch.tensor(req.signal, dtype=torch.float32)
    n_fft = req.n_fft or 2048
    hop_length = req.hop_length or n_fft // 4
    win_length = req.win_length or n_fft
    fs = req.sampling_rate or 12000
    signal_duration_sec = signal.shape[0] / fs
    if n_fft > signal.shape[0]:
        raise HTTPException(
            status_code=400,
            detail=f"n_fft ({n_fft}) must be less than or equal to signal length ({signal.shape[1]})"
        )
    
    temp_dir = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_name = f"spectrogram_{uuid.uuid4().hex}.png"
    file_path = os.path.join(temp_dir, file_name)

    store_stft(signal, n_fft, hop_length, win_length, fs, file_path)
    
    background_tasks.add_task(os.remove, file_path)
    return FileResponse(
        path=file_path,
        filename=file_name,
        media_type="image/png",
    )
    

def store_stft(signal, n_fft, hop_length, win_length, fs, path):
    signal_duration_sec = signal.shape[0] / fs
    x = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(n_fft),
        center=True,
        return_complex=True,
    )

    spectrogram = torch.abs(x)

    spectrogram_np = spectrogram.cpu().numpy()
    
    mag_db_np = 10 * np.log10(spectrogram_np + 1e-6)
    
    freqs_np = torch.linspace(0, fs/2, steps=mag_db_np.shape[0])
    times_np = torch.linspace(0, signal_duration_sec, steps=mag_db_np.shape[1])

    figure, axes = plt.subplots()
    figure.set_size_inches(10, 5)
    axes.set_ylabel('Frequency [Hz]')
    axes.set_xlabel('Time [seconds]')
    axes.pcolormesh(times_np, freqs_np, mag_db_np, shading='gouraud', cmap='magma')
    figure.colorbar(label='Amplitude [dB]', mappable=axes.pcolormesh(times_np, freqs_np, mag_db_np, shading='gouraud', cmap='magma'))
    figure.tight_layout()
    figure.savefig(path)

if __name__ == "__main__":
    import pandas as pd
    sample = pd.read_csv("./samples/sample2_0.007-Ball.csv")
    signal = sample["ch1"].to_numpy()
    store_stft(torch.from_numpy(signal), 2048, 512, 2048, 12000, "test.png")