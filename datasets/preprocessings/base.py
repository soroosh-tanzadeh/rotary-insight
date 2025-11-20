from datasets.dataset import BearingDataset


class BasePreprocessing:
    def __init__(self, train_dataset: BearingDataset, test_dataset: BearingDataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def preprocess(self):
        raise NotImplementedError
