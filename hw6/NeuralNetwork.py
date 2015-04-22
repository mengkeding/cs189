import scipy.io
import numpy as np
import random

import pdb

IMG_SIZE = 28*28
N_SAMPLES = 60000
TEST_SIZE = 10000
VALIDATION_SIZE = 10000

class NeuralNetwork():
    # Initialize Neural Network
    def __init__(self, entropy=False):
        # Layer Sizes
        self.input_layer_size = 784
        self.hidden_layer_size = 200
        self.output_layer_size = 10
        # Weights are None since untrained
        self.W1 = None
        self.W2 = None
        # Uncomment to test backwards prop
        #self.W1 = np.random.randn(self.input_layer_size+1, self.hidden_layer_size)
        #self.W2 = np.random.randn(self.hidden_layer_size+1, self.output_layer_size)

        # Error Function
        # Use Entropy if True else mean squared
        self.entropy = entropy

    """
    X = training images
    Y = training labels
    eta = learning rate
    maxiter = maximum iterations
    """
    def train(self, X, y, eta, maxiter=100000):
        # Weights and bias
        UPPER_BOUND = 1.0
        LOWER_BOUND = -1.0
        self.W1 = (UPPER_BOUND - LOWER_BOUND) * np.random.randn(self.input_layer_size+1, self.hidden_layer_size) + LOWER_BOUND
        self.W2 = (UPPER_BOUND - LOWER_BOUND) * np.random.randn(self.hidden_layer_size+1, self.output_layer_size) + LOWER_BOUND


        # Pad for bias
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        y_labels = y
        # Transform Y
        transform_y = lambda digit: np.identity(10)[digit]
        tmp = []
        for i in range(len(y)):
            tmp.append(transform_y(y[i]))
        y = np.array(tmp)

        for iteration in range(maxiter):
            # Random index to split training set for validation
            self.randomIndex = np.random.choice(N_SAMPLES, N_SAMPLES, replace=False)
            # Split Data
            train_images = X[self.randomIndex[:-VALIDATION_SIZE]]
            train_labels = y[self.randomIndex[:-VALIDATION_SIZE]]
            validation_images = X[self.randomIndex[-VALIDATION_SIZE:]]
            validation_labels = y[self.randomIndex[-VALIDATION_SIZE:]]
            y_labels = y_labels[net.randomIndex[:]]

            # Forward Pass
            _ = self.forward(train_images)
            # Calculate cost/loss
            cost = self.costFunction(train_images, train_labels)
            #pdb.set_trace()
            # G: input gradient      g: output gradient
            G, g = self.backwards(train_images, train_labels)
            self.W1 -= eta*G
            self.W2 -= eta*g
            # For Testing
            #if iteration % 1000 == 0:
            print "Iteration: %d" % (iteration)
            predicted = self.predict(X)
            incorrect = np.count_nonzero(y_labels - predicted)
            accuracy = (predicted.shape[0] - incorrect) / float(predicted.shape[0])
            print "Training Accuracy: %f" % accuracy
            #print "Iteration: %d" % (iteration)
        return self.W1, self.W2

    def predict(self, X):
        if (self.W1 is not None) and (self.W2 is not None):
            probability = self.forward(X)
            labels = np.argmax(probability, axis=1)
            return labels
        else:
            return None

    def forward(self, X):
        # Propagate inputs forward through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.tanh(self.z2) #Hidden Layer Activation
        self.a2 = np.hstack((self.a2, np.ones((self.a2.shape[0], 1))))
        self.z3 = np.dot(self.a2, self.W2)
        self.yHat = self.sigmoid(self.z3)
        return self.yHat

    #TODO: Fix this shit here
    def backwards(self, X, y):
        #self.yHat = self.forward(X)
        # Output to Hidden
        delta3 = np.multiply(-(y - self.yHat), self.sigmoid_derivative(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        # Hidden to Input
        delta2 = np.dot(delta3, self.W2[:-1].T) * self.tanh_derivative(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2

    def costFunction(self, X, y, flag=False):
        if flag == True:
            self.yHat = self.forward(X)
        cost_vector = np.array([])
        for i in range(len(y)):
            J = 0.5 * np.sum(np.square(y[i] - self.yHat[i]))
            cost_vector = np.append(cost_vector, J)
        return cost_vector

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

####################################################
DATA_DIR="/Users/David/dev/cs189/hw6/digit-dataset/"
train_data = scipy.io.loadmat(DATA_DIR+"train.mat")
test_data = scipy.io.loadmat(DATA_DIR+"test.mat")

# Load Data
X = train_data['train_images'].transpose([2,0,1]).ravel().reshape((N_SAMPLES, IMG_SIZE)).astype(float)
Y = train_data['train_labels'].ravel()

# TODO: Pick preprocessing
xTrain = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)
yTrain = Y
#X /= 256.0
import pdb; pdb.set_trace()


# Create Neural Network
net = NeuralNetwork()
net.train(xTrain, yTrain, 1e-9, maxiter=10000)
#validate_labels = net.predict(xValidate)
pdb.set_trace()
#net.backwards(xTrain, yTrain)
