import numpy as np
import random

class NeuralNetwork():
    # Initialize Neural Network
    def __init__(self):
        # Layer Sizes
        self.input_layer_size = 784
        self.hidden_layer_size = 200
        self.output_layer_size = 10
        # Weights are None since untrained
        #self.W1 = None
        #self.W2 = None
        # Uncomment to test backwards prop
        self.W1 = np.random.randn(self.input_layer_size+1, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size+1, self.output_layer_size)

    """
    X = training images
    Y = training labels
    eta = learning rate
    maxiter = maximum iterations
    """
    def train(self, X, y, eta, maxiter=10000):
        # Weights and bias
        self.W1 = np.random.randn(self.input_layer_size+1, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size+1, self.output_layer_size)
        for iteration in range(maxiter):
            import pdb; pdb.set_trace()
            yHat = self.forward(X)
            G, g = self.backwards(X, y)
            self.W1 -= eta*G
            self.W2 -= eta*g
            print "Iteration: %d" % iteration
        return self.W1, self.W2

    def predict(self, X):
        if (self.W1 is not None) and (self.W2 is not None):
            return forward(X)
        else:
            return None

    def forward(self, X):
        # Propagate inputs forward through network
        self.z2 = np.dot(X, self.W1[:-1])
        self.a2 = self.tanh(self.z2 + np.ones(self.z2.shape) * self.W1[-1]) #Hidden Layer Activation
        self.z3 = np.dot(self.a2, self.W2[:-1])
        self.yHat = self.sigmoid(self.z3 + np.ones(self.z3.shape) * self.W2[-1])
        return self.yHat

    def backwards(self, X, y):
        import pdb; pdb.set_trace()
        error = self.costFunction(X, y)
        # hidden layer
        tmp = np.dot(np.ones((self.a2.shape[0],1)), error)
        g = -np.dot(tmp.T, self.a2).T
        #TODO: fix dimensions here
        error = error + np.dot(error, self.W2)*self.tanh_derivative(self.z2)
        # input layer
        G = -np.dot(error, self.a2)
        return G, g


    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat)**2)
        return np.array([J])

    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        # Output to Hidden
        delta3 = np.multiply(-(y - self.yHat), self.sigmoid_derivative(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        # Hidden to Input
        delta2 = np.dot(delta3, self.W2.T) * self.tanh_derivative(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2

    def crossEntropy(self, X, y):
        self.yHat = self.forward(X)
        term1 = y*np.log(self.yHat)
        term2 = (np.ones(y.shape)-y) * np.log(np.ones(self.yHat.shape) - self.yHat)
        total = sum(term1 + term2)
        J = -total
        #J = -sum(y*np.log(self.yHat) + (np.ones(y.shape)-y)*np.log(np.ones(self.yHat.shape)-self.yHat))
        return J

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def tanh_derivative(self, x):
        return 1 - self.tanh(x)**2

import scipy.io
import numpy as np

DATA_DIR="/Users/David/dev/cs189/hw6/digit-dataset/"
train_data = scipy.io.loadmat(DATA_DIR+"train.mat")
test_data = scipy.io.loadmat(DATA_DIR+"test.mat")

IMG_SIZE = 28*28
N_SAMPLES = 60000
TEST_SIZE = 10000
VALIDATION_SIZE = 10000
TRAIN_SIZE = N_SAMPLES - VALIDATION_SIZE

# Load Data
#X = np.matrix(train_data['train_images'].transpose([2,0,1]).ravel().reshape((N_SAMPLES, IMG_SIZE))).astype(float)
#Y = np.matrix(train_data['train_labels'].ravel()).T
X = train_data['train_images'].transpose([2,0,1]).ravel().reshape((N_SAMPLES, IMG_SIZE)).astype(float)
Y = train_data['train_labels'].ravel().T

# TODO: Pick preprocessing
#X = (X - X.mean()) / X.std()
X /= 256.0

# Random index to split training set for validation
randomIndex = np.random.choice(N_SAMPLES, N_SAMPLES, replace=False)

def transform_y(value):
    identity = np.identity(10)
    return identity[value]

# Split Data
xTrain = X[randomIndex[:-VALIDATION_SIZE]]
yTrain = Y[randomIndex[:-VALIDATION_SIZE]]
xValidate = X[randomIndex[-VALIDATION_SIZE:]]
yValidate = Y[randomIndex[-VALIDATION_SIZE:]]
xTest = np.matrix(test_data['test_images'].transpose([2,0,1]).ravel().reshape((TEST_SIZE, IMG_SIZE)))

# Reformat y values to be vectors
tmp = []
for i in range(len(yTrain)):
    tmp.append(transform_y(yTrain[i]))
yTrain = np.array(tmp)

tmp = []
for i in range(len(yValidate)):
    tmp.append(transform_y(yValidate[i]))

# Create Neural Network
net = NeuralNetwork()
#net.train(xTrain, yTrain, 3, maxiter=10)
net.backwards(xTrain, yTrain)
