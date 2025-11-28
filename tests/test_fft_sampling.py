import numpy as np
import pytest
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

def test_fft_sampling():
    # Create a signal: 100Hz sine wave sampled at 1000Hz
    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)
    signal = np.sin(2 * np.pi * 100 * t).tolist()
    
    # Use a valid API key (mocked or from env)
    # Since we are using TestClient, we can override dependency or use a known key if set in env.
    # For now, let's assume 'test' works or we need to set env var.
    api_key = "test"
    headers = {"X-API-Key": api_key}

    # Test 1: No sampling rate (should be normalized)
    payload_no_fs = {
        "signal": signal,
        "n": 1000
    }
    
    resp = client.post("/processing/fft", json=payload_no_fs, headers=headers)
    assert resp.status_code == 200, f"Test 1 Failed: {resp.text}"
    
    data = resp.json()
    freqs = data['frequencies']
    # Peak should be at 0.1 (100/1000)
    peak_idx = np.argmax(data['magnitudes'])
    peak_freq = freqs[peak_idx]
    print(f"Test 1 (No FS): Peak freq = {peak_freq} (Expected ~0.1)")
    assert abs(peak_freq - 0.1) < 0.01

    # Test 2: With sampling rate (should be in Hz)
    payload_with_fs = {
        "signal": signal,
        "n": 1000,
        "sampling_rate": fs
    }
    
    resp = client.post("/processing/fft", json=payload_with_fs, headers=headers)
    assert resp.status_code == 200, f"Test 2 Failed: {resp.text}"
    
    data = resp.json()
    freqs = data['frequencies']
    # Peak should be at 100
    peak_idx = np.argmax(data['magnitudes'])
    peak_freq = freqs[peak_idx]
    print(f"Test 2 (With FS={fs}): Peak freq = {peak_freq} (Expected ~100.0)")
    assert abs(peak_freq - 100.0) < 1.0

if __name__ == "__main__":
    test_fft_sampling()
