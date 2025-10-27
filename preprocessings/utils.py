from sklearn.preprocessing import StandardScaler
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


def scale_data(train_data, test_data):
    """
    Scales the training and testing data using Min-Max scaling.

    Args:
        train_data (np.ndarray): The training data.
        test_data (np.ndarray): The testing data.

    Returns:
        tuple: Scaled training and testing data.
    """
    scaler = StandardScaler()
    num_samples, num_channels, signal_length = train_data.shape
    train_data_reshaped = train_data.reshape(num_samples, -1)
    test_data_reshaped = test_data.reshape(test_data.shape[0], -1)

    scaler.fit(train_data_reshaped)
    train_data_scaled = scaler.transform(train_data_reshaped).reshape(
        num_samples, num_channels, signal_length
    )
    test_data_scaled = scaler.transform(test_data_reshaped).reshape(
        test_data.shape[0], num_channels, signal_length
    )

    return train_data_scaled, test_data_scaled
