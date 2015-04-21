import numpy as np
import random

class NeuralNetwork():
    # Initialize Neural Network
    def __init__(self, sizes):
        self.input_layer_size = 784
        self.hidden_layer_size = 200
        self.output_layer_size = 10

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def sigmoid_derivative(self, x):
        return sigmoid(x) * (1.0 - sigmoid(x))

    def tanh_derivative(self, x):
        return 1 - tanh(x)**2

    #def forward_pass(self, a):
    #    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    #    for bias, weight in zip(self.biases, self.weights):
    #        a = np.vectorize(sigmoid)(np.dot(weight, a) + bias)
    #    return a

    #def SGD(self, xTrain, yTrain):
    #    training_data = [ (xTrain[i], yTrain[i]) for i in range(len(yTrain)) ] # list of (x, y)

import pdb; pdb.set_trace()
