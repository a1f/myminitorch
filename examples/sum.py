import typing as t
from attr import dataclass
import numpy as np
from dataloader import DataLoader
from dataset import Dataset

from layer import *
from nn import NeuralNetwork
from optimizer import SGD
from tensor import Tensor
from trainer import train


# TODO: change to 8
BITS_IN_NUMBER = 2
HIDDEN_LAYER = BITS_IN_NUMBER
LR = 0.01
DATASET_SIZE = 10000


@dataclass
class DataItem:
    num1: int
    num2: int
    result: int


def prepare_data() -> t.List[DataItem]:
    result = []
    for _ in range(DATASET_SIZE):
        x = np.random.randint(0, 1**BITS_IN_NUMBER)
        y = np.random.randint(0, 1**BITS_IN_NUMBER)
        result.append(DataItem(
            num1=x,
            num2=y,
            result=x+y,
        ))
    return result


class SumDataset(Dataset):

    def __init__(self, data: t.List[DataItem]) -> None:
        self._raw_data = data

    @staticmethod
    def int_to_tensor(x: int, width: int = BITS_IN_NUMBER) -> Tensor:
        return np.array(list(np.binary_repr(x, width)), dtype=int)

    def __getitem__(self, index: int) -> t.Tuple[Tensor, Tensor]:
        data_instance = self._raw_data[index]
        return (
            np.concatenate(
                SumDataset.int_to_tensor(data_instance.num1),
                SumDataset.int_to_tensor(data_instance.num2),
            ), 
            SumDataset.int_to_tensor(data_instance.result, BITS_IN_NUMBER + 1),
        )

    def __len__(self) -> int:
        return len(self._raw_data)
    

print("Initialize neural network...")
nn = NeuralNetwork((
    Linear(BITS_IN_NUMBER + BITS_IN_NUMBER, HIDDEN_LAYER),
    ReLU(),
    Linear(HIDDEN_LAYER, BITS_IN_NUMBER + 1),
    Sigmoid(),
))
sgd_optimizer = SGD(nn, LR)

print("Preparing data...")
raw_data = prepare_data()
len_raw_data = len(raw_data)
train_size = int(0.7 * len_raw_data)
train_dataset = SumDataset(raw_data[:train_size])
test_dataset = SumDataset(raw_data[train_size:])

train_data_loader = DataLoader(train_dataset)
validation_data_loader = DataLoader(test_dataset)

print("Beginning to train and validating!")
train(nn, train_data_loader, sgd_optimizer, validator_data_loader=validation_data_loader)
    
print("Want to inference?")
# TODO: inference