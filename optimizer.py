import typing as t
import numpy as np

from layer import *
from nn import NeuralNetwork
from tensor import Tensor


class Optimizer(ABC):
    def __init__(self, nn: NeuralNetwork, lr: float) -> None:
        self._nn = nn
        self._lr = lr

    @abstractmethod
    def step(self) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, nn: NeuralNetwork, lr: float) -> None:
        super().__init__(nn, lr)

    def step(self) -> None:
        for param, grad in self._nn.params_and_grads():
            param -= self._lr * grad
    