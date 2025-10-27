import torch
from datasets.dataset import BearingDataset
from datasets.cwru_dataset import CrwuDataset
from datasets.pu_dataset import PUDataset


class Experiment:
    def __init__(
        self,
        name: str,
        model: str,
        dataset: BearingDataset,
        test_with_noise: bool = False,
        train_with_noise: bool = False,
        train_augmentation: bool = False,
        method: str = "train_validation",
        noise_level: float = None,
        trials: int = 1,
    ):
        if method not in ["train_validation", "cross_validation"]:
            raise ValueError(
                "Scoring method must be 'train_validation' or 'cross_validation'"
            )

        self.name = name
        self.model = model
        self.test_with_noise = test_with_noise
        self.train_with_noise = train_with_noise
        self.train_augmentation = train_augmentation
        self.method = method
        self.dataset = dataset
        self.noise_level = noise_level
        self.trials = trials

    def get_dataset(self) -> BearingDataset:
        return self.dataset


def dataset(name, experiment_name, window_size=2048, device=None) -> BearingDataset:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if name == "CWRU":
        ds = CrwuDataset(
            rdir="./data/dataset/CWRU/",
            fault_location="DriveEnd",
            seq_len=window_size,
            rpms=["1797", "1772", "1750", "1730"],
        )

        return ds

    elif name == "PU":
        ds = PUDataset(
            rdir="./data/dataset/PU/",
            window_size=window_size,
            step_size=window_size,
        )

        return ds
    else:
        raise ValueError(f"Dataset {name} not supported")


def create_experiments(
    name: str,
    test_with_noise: bool = False,
    train_with_noise: bool = False,
    train_augmentation: bool = False,
    method: str = "train_validation",
    noise_levels: list = None,
    models: list = None,
    trials: int = 1,
    dataset_name: str = "cwru",
    window_size: int = 2048,
    device=None,
) -> list[Experiment]:
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

    if train_with_noise and train_augmentation:
        raise ValueError(
            "Cannot train with noise and augment training data at the same time"
        )

    experiments = []

    for noise_level in noise_levels:
        for model_name in models:

            ds = dataset(
                name=dataset_name,
                experiment_name=name,
                window_size=window_size,
                device=device,
            )

            experiment_config = Experiment(
                name=name,
                model=model_name,
                test_with_noise=test_with_noise,
                train_augmentation=train_augmentation,
                train_with_noise=train_with_noise,
                method=method,
                noise_level=noise_level,
                trials=trials,
                dataset=ds,
            )
            experiments.append(experiment_config)

    return experiments
