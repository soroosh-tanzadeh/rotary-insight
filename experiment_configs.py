# experiment_configs.py
"""
Configuration module for creating and managing experiments.
"""


def create_experiments(
    name: str,
    test_with_noise: bool = False,
    train_augmentation: bool = False,
    scoring_method: str = "train_validation",
    noise_levels: list = None,
    models: list = None,
    number_of_random_splits: int = 1,
) -> list:
    """
    Creates experiment configurations based on provided parameters.

    Args:
        name: Experiment name
        test_with_noise: Whether to test with noisy data
        train_augmentation: Whether to augment training data with noise
        scoring_method: Scoring method - "train_validation" or "cross_validation"
        noise_levels: List of noise levels in SNR dB
        models: List of model names to train
        number_of_random_splits: Number of random train/test splits

    Returns:
        List of experiment configurations
    """
    if noise_levels is None:
        noise_levels = [0.0]

    if models is None:
        models = ["transformer_encoder_classifier"]

    experiments = []

    for noise_level in noise_levels:
        for model_name in models:
            experiment_config = {
                "name": f"{name}_{model_name}_noise_{noise_level}",
                "model_name": model_name,
                "scoring_method": scoring_method,
                "test_with_noise": test_with_noise,
                "train_augmentation": train_augmentation,
                "test_noise_config": {"snr_dB": noise_level},
                "train_noise_config": {"snr_dB": noise_level},
                "number_of_random_splits": number_of_random_splits,
                # Default CWRU experiment parameters
                "exps": ["12DriveEndFault"],
                "rpms": ["1797"],
            }
            experiments.append(experiment_config)

    return experiments
