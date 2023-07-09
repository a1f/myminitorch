import unittest
from unittest import TestCase
from layer import Linear, ReLU, Sigmoid
import numpy as np
from tensor import Tensor


class TestLayer(TestCase):

    def test_linear_layer_one_neuron(self) -> None:
        linear_layer = Linear(1, 1, True)
        w = np.array([[2]])
        b = np.array([[1]])
        linear_layer.set(w=w, bias=b)
        x = np.array([[-1]])

        # -1 * 2 + 1 = -1
        value = linear_layer.forward(x)
        self.assertEquals(value, -1)
        res = linear_layer.backward(np.array([5]))
        self.assertEqual(
            linear_layer._grads["w"],
            np.array([-5]),
        )
        self.assertEqual(
            linear_layer._grads["bias"],
            np.array([5])
        )
        self.assertEqual(res, np.array([10]))

    def test_linear_layer_one_neuron_batch_of_2_samples(self) -> None:
        linear_layer = Linear(1, 1, True)
        w = np.array([[2]])
        b = np.array([[1]])
        linear_layer.set(w=w, bias=b)
        x = np.array([[-2, 3]])

        """
        W is 2, B is 1, X is (-2, 3) - 2 samples
        W @ X => (-4 6) + B => (-3 7)
        backprop
        dG = (-1 3)
        dG / dW = 11
        
        """

        # -1 * 2 + 1 = -1
        value = linear_layer.forward(x)
        self.assertTrue(np.array_equal(value, np.array([[-3, 7]])), f"different values {value} vs [[-3, 7]]")
        res = linear_layer.backward(np.array([[-1, 3]]))
        self.assertTrue(np.array_equal(linear_layer._grads["w"], np.array([[11]])))
        self.assertTrue(np.array_equal(linear_layer._grads["bias"], np.array([[2]])))
        self.assertTrue(np.array_equal(res, np.array([[-2, 6]])))

    def test_linear_layer_full_sample(self) -> None:
        """
        2 samples, 3 inputs, 2 outputs
        W = 1 2 3  B = -1  X = 1 2
            4 5 6      -2      1 3
                               2 1

        W @ X = 9 11 => + B = 8 10
               21 29         19 27

        dG = -1 -2
              2 -1
        """
        linear_layer = Linear(3, 2, True)
        w = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])
        b = np.array([[-1], [-2]])
        linear_layer.set(w=w, bias=b)
        x = np.array([
            [1, 2],
            [1, 3],
            [2, 1],
        ])

        value = linear_layer.forward(x)
        self.assertTrue(np.array_equal(value, np.array(
            [[8, 10], [19, 27]]
        )))
        res = linear_layer.backward(np.array([[-1, -2], [2, -1]]))
        self.assertTrue(np.array_equal(linear_layer._grads["w"], np.array([
            [-5, -7, -4],
            [0, -1, 3],
        ])), f"Different result for dG/dW: {linear_layer._grads['w']}")
        self.assertTrue(np.array_equal(linear_layer._grads["bias"], np.array([[-3], [1]])), 
                        f"Different result for dG/db: {linear_layer._grads['bias']}")
        self.assertTrue(np.array_equal(res, np.array([
            [7, -6],
            [8, -9],
            [9, -12],
        ])))

    def test_activation_layer_relu(self) -> None:
        input_tensor = np.array([[-1.0, 0.0], [1.0, 2.0]])
        expected_output_tensor = np.array([[0.0, 0.0], [1.0, 2.0]])

        relu = ReLU()
        # Forward pass
        output_tensor = relu.forward(input_tensor)
        self.assertTrue(np.array_equal(output_tensor, expected_output_tensor))

        # Backward pass
        grad = np.array([[2.0, 3.0], [4.0, 0.0]])
        result = relu.backward(grad)
        expected_input_grad_tensor = np.array([[0.0, 0.0], [4.0, 0.0]])
        self.assertTrue(np.array_equal(result, expected_input_grad_tensor), f"got {result}")

    def test_activation_layer_sigmoid(self) -> None:
        input_tensor = np.array([[-1.0, 0.0], [1.0, 2.0]])
        expected_output_tensor = [[0.26894142, 0.5], [0.73105858, 0.88079708]]

        sigm = Sigmoid()
        # Forward pass
        output_tensor = sigm.forward(input_tensor)
        self.assertTrue(np.allclose(output_tensor, expected_output_tensor), f"got {output_tensor}")

        # Backward pass
        grad = np.array([[2.0, 3.0], [4.0, 0.0]])
        result = sigm.backward(grad)
        expected_input_grad_tensor = np.array([[0.39322386, 0.75], [0.78644772, 0.0]])
        self.assertTrue(np.allclose(result, expected_input_grad_tensor), f"got {result}")


# Run the tests
if __name__ == '__main__':
    unittest.main()