from abc import ABC, abstractmethod

import numpy as np
from tensor import Tensor


class Loss(ABC):
    @abstractmethod
    def loss(self, y: Tensor, y_hat: Tensor) -> float:
        pass

    @abstractmethod
    def grad(self, y: Tensor, y_hat: Tensor) -> Tensor:
        pass


class MSELoss(Loss):
    def loss(self, y: Tensor, y_hat: Tensor) -> float:
        return np.sum((y - y_hat)**2) / 2

    def grad(self, y: Tensor, y_hat: Tensor) -> Tensor:
        return y - y_hat