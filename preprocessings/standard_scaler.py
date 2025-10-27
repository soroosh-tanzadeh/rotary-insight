from preprocessings.base import BasePreprocessing
from datasets.dataset import BearingDataset

class StandardScaler(BasePreprocessing):
    def __init__(self, train_dataset: BearingDataset, test_dataset: BearingDataset):
        super().__init__(train_dataset, test_dataset)
        self.mu = None
        self.sigma = None

    def preprocess(self):
        self.mu = self.train_dataset.inputs().mean(axis=0)
        self.sigma = self.train_dataset.inputs().std(axis=0)
        self.train_dataset.set_inputs(
            (self.train_dataset.inputs() - self.mu) / self.sigma
        )
        self.test_dataset.set_inputs(
            (self.test_dataset.inputs() - self.mu) / self.sigma
        )
        return self.train_dataset, self.test_dataset

    def __repr__(self):
        return f"StandardScaler(mu={self.mu}, sigma={self.sigma})"
