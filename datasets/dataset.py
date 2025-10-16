import torch
from abc import abstractmethod, ABC


class BearingDataset(torch.utils.data.Dataset, ABC):
    @abstractmethod
    def labels(self) -> list[str]:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def window_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def inputs(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def targets(self) -> torch.Tensor:
        raise NotImplementedError


class SimpleBearingDataset(BearingDataset):
    def __init__(self, X, y, labels, window_size):
        self.X = X
        self.y = y
        self.classes = labels
        self.window_size = window_size

        assert X.shape[0] == y.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def inputs(self) -> torch.Tensor:
        return self.X

    def targets(self) -> torch.Tensor:
        return self.y

    def labels(self) -> list[str]:
        return self.classes

    def window_size(self) -> int:
        return self.window_size

    def __repr__(self):
        return f"SimpleBearingDataset(X={self.X.shape}, y={self.y.shape}, labels={self.classes}, window_size={self.window_size})"
