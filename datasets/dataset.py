import torch
import numpy as np
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
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def stack(self, X: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def set_inputs(self, X: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def set_targets(self, y: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def shuffle(self):
        raise NotImplementedError

    @abstractmethod
    def inputs(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def targets(self) -> torch.Tensor:
        raise NotImplementedError


class SimpleBearingDataset(BearingDataset):
    def __init__(
        self, X: torch.Tensor, y: torch.Tensor, labels: list[str], window_size: int
    ):
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

    def to(self, device: torch.device) -> "SimpleBearingDataset":
        self.X = self.X.to(device)
        self.y = self.y.to(device)
        return self

    def shuffle(self):
        indices = torch.arange(len(self.X))
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]
        return self

    def stack(self, X: torch.Tensor, y: torch.Tensor) -> "SimpleBearingDataset":
        self.X = torch.vstack((self.X, X))
        self.y = torch.hstack((self.y, y))
        return self

    def set_inputs(self, X: torch.Tensor) -> "SimpleBearingDataset":
        self.X = X
        return self

    def set_targets(self, y: torch.Tensor) -> "SimpleBearingDataset":
        self.y = y
        return self

    def window_size(self) -> int:
        return self.window_size

    def __repr__(self):
        return f"SimpleBearingDataset(X={self.X.shape}, y={self.y.shape}, labels={self.classes}, window_size={self.window_size})"
