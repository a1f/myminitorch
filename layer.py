from abc import ABC, abstractmethod

import numpy as np
import typing as t
from tensor import Tensor


class Layer(ABC):

    def __init__(self) -> None:
        self._parameters: t.Dict[str, Tensor] = {}
        self._grads: t.Dict[str, Tensor] = {}
        self._x: t.Optional[Tensor] = None

    @abstractmethod
    def set(self, **kwards: t.Dict[str, t.Any]) -> None:
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass 

    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor:
        pass

    def zero_grad(self) -> None:
        for key in self._grads:
            self._grads[key] = np.zeros_like(self._grads[key])


class Linear(Layer):

    """
    Linear layer provides transformation from X to W * X + B.
    Dimentions:
        W is (n_output, n_input)
        X is (n_input, n_samples)
        B is (n_output, 1)

        W @ X + B, note + is a broadcast

        output is (n_output, n_samples)
    """

    p_w: t.ClassVar[str] = "w"
    p_b: t.ClassVar[str] = "bias" 

    def __init__(self, n_input: int, n_output: int, bias: bool = True) -> None:
        super().__init__()
        self._parameters[Linear.p_w] = np.zeros((n_output, n_input))
        self._parameters[Linear.p_b] = np.zeros((n_output, 1))

    def set(self, **kwargs: t.Any) -> None:
        for key, value in kwargs.items():
            assert key in self._parameters, f"{key} is not presented in parameters"
            self._parameters[key] = value

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self._x = x
        return self._parameters[Linear.p_w] @ x + self._parameters[Linear.p_b]

    def backward(self, grad: Tensor) -> Tensor:
        if grad.ndim == 1:
            grad = grad.reshape(-1, 1)
        
        grad_for_batch = np.sum(grad, axis=1, keepdims=True)
        self._grads[Linear.p_b] = grad_for_batch
        assert self._x is not None, "At least one forward iteration should has been run"
        print(f"{grad_for_batch=}")
        print(f"{self._x.T=}")

        # grad should be (n_output, n_samples)
        # w should be (n_output, n_inputs)
        # X is (n_input, n_samples)

        self._grads[Linear.p_w] = grad @ self._x.T
        return self._parameters[Linear.p_w].T @ grad


F = t.Callable[[Tensor], Tensor]


class ActivationLayer(Layer, ABC):
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self._f = f
        self._f_prime = f_prime

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self._x = x
        return self._f(x)
    
    def backward(self, grad: Tensor) -> Tensor:
        if grad.ndim == 1:
            grad = grad.reshape(-1, 1)
        assert self._x is not None
        return self._f_prime(self._x) * grad


class ReLU(ActivationLayer):
    def __init__(self) -> None:
        super().__init__(lambda x: np.maximum(0, x), lambda x: np.where(x > 0, 1, 0))

    def set(self, **kwargs: t.Any) -> None:
        raise NotImplementedError("We don't set anything for ReLU")

class Sigmoid(ActivationLayer):
    def __init__(self) -> None:
        sigm = lambda x: 1 / (1 + np.exp(-x))
        super().__init__(sigm, lambda x: sigm(x) * (1 - sigm(x)))

    def set(self, **kwargs: t.Any) -> None:
        raise NotImplementedError("We don't set anything for Sigmoid")