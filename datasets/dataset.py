import torch
from abc import abstractmethod


class BearingDataset(torch.utils.data.Dataset):
    @abstractmethod
    def labels(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def window_size(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def inputs(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def targets(self) -> torch.Tensor:
        raise NotImplementedError
