import numpy as np
import random

class NeuralNetwork():
    # Initialize Neural Network
    def __init__(self):
        # Layer Sizes
        self.input_layer_size = 784
        self.hidden_layer_size = 200
        self.output_layer_size = 10

        # Weights
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)

    def forward(self, X):
        # Propagate inputs forward through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.tanh(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.yHat = self.sigmoid(self.z3)
        return self.yHat

    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        # Output to Hidden
        delta3 = np.multiply(-(y - self.yHat), self.sigmoid_derivative(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        # Hidden to Input
        delta2 = np.dot(delta3, self.W2.T) * self.tanh_derivative(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def sigmoid_derivative(self, x):
        return sigmoid(x) * (1.0 - sigmoid(x))

    def tanh_derivative(self, x):
        return 1 - tanh(x)**2

import pdb; pdb.set_trace()
