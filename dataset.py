import typing as t
from tensor import Tensor 
from abc import ABC, abstractmethod
 

class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, index: int):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
