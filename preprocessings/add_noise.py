from preprocessings.base import BasePreprocessing
from datasets.dataset import BearingDataset
from preprocessings.utils import add_noise_data


class AddNoise(BasePreprocessing):
    def __init__(
        self,
        train_dataset: BearingDataset,
        test_dataset: BearingDataset,
        noise_level: float,
        to_training: bool = True,
        to_test: bool = False,
    ):
        super().__init__(train_dataset, test_dataset)
        self.noise_level = noise_level
        self.to_training = to_training
        self.to_test = to_test

    def preprocess(self):
        if self.to_training:
            self.train_dataset.set_inputs(
                add_noise_data(self.train_dataset.inputs(), self.noise_level)
            )
        if self.to_test:
            self.test_dataset.set_inputs(
                add_noise_data(self.test_dataset.inputs(), self.noise_level)
            )
        self.train_dataset.shuffle()
        return self.train_dataset, self.test_dataset
