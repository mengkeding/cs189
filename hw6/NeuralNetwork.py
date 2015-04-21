import numpy as np
import random

class NeuralNetwork():
    # Initialize Neural Network
    """
    Sizes is a list of sizes for each layer
    Ex. neural_network = NeuralNetwork([784, 200, 10])
    784 input, 200 hidden, 10 output
    randomizes biases and weights on initialization
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def forward_pass(self, a):
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
        for bias, weight in zip(self.biases, self.weights):
            a = np.vectorize(sigmoid)(np.dot(weight, a) + bias)
        return a

    def SGD(self, xTrain, yTrain):
        training_data = [ (xTrain[i], yTrain[i]) for i in range(len(yTrain)) ] # list of (x, y)

tmp = NeuralNetwork([784, 200, 10])
import pdb; pdb.set_trace()
