import torch


def add_noise_data(signal, noise_level):
    """
    Adds Gaussian noise to the signal based on the specified noise level.

    Args:
        signal (torch.Tensor): The input signal.
        noise_level (float): The noise level in dB.

    Returns:
        torch.Tensor: The noisy signal.
    """
    snr = 10 ** (noise_level / 10)
    signal_power = torch.mean(signal**2)
    noise_power = signal_power / snr
    noise = torch.normal(0, torch.sqrt(noise_power), signal.shape)
    return signal + noise
