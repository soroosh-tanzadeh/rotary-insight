from preprocessings.base import BasePreprocessing
from preprocessings.utils import add_noise_data
from datasets.dataset import BearingDataset


class TrainAugmentation(BasePreprocessing):
    def __init__(
        self,
        train_dataset: BearingDataset,
        test_dataset: BearingDataset,
        noise_level: float,
    ):
        super().__init__(train_dataset, test_dataset)
        self.noise_level = noise_level

    def preprocess(self):
        X_train_noisy = add_noise_data(self.train_dataset.inputs(), self.noise_level)
        self.train_dataset.stack(X_train_noisy, self.train_dataset.targets())
        self.train_dataset.shuffle()
        return self.train_dataset, self.test_dataset
