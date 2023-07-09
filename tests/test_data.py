import unittest
from unittest import TestCase
from dataloader import DataLoader
from dataset import Dataset
from layer import Linear, ReLU, Sigmoid
import numpy as np
from tensor import Tensor


class TestDataset(Dataset):
    def __init__(self, inputs: Tensor, targets: Tensor) -> None:
        self._inputs = inputs
        self._targets = targets

    def __getitem__(self, index: int):
        return self._inputs[index], self._targets[index]
    
    def __len__(self) -> int:
        return len(self._inputs)
    

class TestData(TestCase):
    """Unit tests that covers Dataset and DataLoader classes."""

    def test_data_loader_with_custom_dataset(self) -> None:
        np.random.seed(42)
        test_inputs: np.ndarray = np.random.randn(5, 5)
        test_outputs: np.ndarray = np.random.randn(5, 2)
        this_dataset = TestDataset(test_inputs, test_outputs)
        data_loader = DataLoader(this_dataset, batch_size=3, shuffle=False)
        print("here")
        print(data_loader._dataset, data_loader._batch_size)
        expected_batch_sizes = [3, 2]
        count_batches = 0
        current_sample = 0
        for batch in data_loader:
            self.assertEqual(len(batch), expected_batch_sizes[count_batches])
            for sample in batch:
                input_sample, output_sample = sample
                self.assertTrue(np.allclose(test_inputs[current_sample], input_sample))
                self.assertTrue(np.allclose(test_outputs[current_sample], output_sample))
                current_sample += 1
            count_batches += 1
        self.assertEqual(2, count_batches)
        self.assertEqual(5, current_sample)


# Run the tests
if __name__ == '__main__':
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestData)
    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)