from abc import ABC, abstractmethod

import numpy as np
from dataset import Dataset


class DataLoaderIterator:
    def __init__(self, dataset: Dataset, batch_size: int, starts: np.ndarray) -> None:
        self._len = len(dataset)
        self._cur = 0
        self._dataset = dataset
        self._batch_size = batch_size
        self._starts = starts

    def __next__(self):
        if self._cur >= self._len:
            raise StopIteration
        right_edge = min(self._cur + self._batch_size, self._len)
        batch_data = [self._dataset[idx] for idx in range(self._cur, right_edge)]
        self._cur += self._batch_size
        return batch_data


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._starts = np.arange(0, len(dataset), self._batch_size)
        if self._shuffle:
            np.random.shuffle(self._starts)

    def reshuffle(self) -> None:
        np.random.shuffle(self._starts)

    def __iter__(self) -> DataLoaderIterator:
        return DataLoaderIterator(self._dataset, self._batch_size, self._starts)
    
