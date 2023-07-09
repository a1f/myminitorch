import typing as t

from layer import Layer
from tensor import Tensor


class NeuralNetwork:
    def __init__(self, layers: t.Sequence[Layer]) -> None:
        self._layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self._layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self._layers):
            grad = layer.backward(grad)
        return grad
    
    def params_and_grads(self) -> t.Iterator[t.Tuple[Tensor, Tensor]]:
        for layer in self._layers:
            for name, param in layer._parameters.items():
                yield param, layer._grads[name]

    def zero_grads(self) -> None:
        for layer in self._layers:
            layer.zero_grad()