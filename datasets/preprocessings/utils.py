import numpy as np


def add_noise_data(signal, noise_level):
    """
    Adds Gaussian noise to the signal based on the specified noise level.

    Args:
        signal (np.ndarray): The input signal.
        noise_level (float): The noise level in dB.

    Returns:
        np.ndarray: The noisy signal.
    """
    snr = 10 ** (noise_level / 10)
    signal_power = np.mean(signal**2)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise
