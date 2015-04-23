import scipy.io
import numpy as np

import pdb

IMG_SIZE = 28*28
N_SAMPLES = 60000
TEST_SIZE = 10000
VALIDATION_SIZE = 50000

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
        # Learning Rate
        self.eta = None
        # TODO: below
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
        self.eta = eta

        # Pad for bias
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        y_labels = y
        # Transform Y
        def transform_y(digit):
            vector = np.zeros(10)
            vector[digit] = 1
            return vector

        for iteration in range(maxiter):
            print "Iteration: %d" % (iteration)

            # Random index to split training set for validation
            self.randomIndex = np.random.choice(X.shape[0], X.shape[0], replace=False)

            for i in self.randomIndex[:10000]:
                ## Select a data point
                image = X[i]
                label = transform_y(y[i])
                # Forward Pass
                hidden, prediction = self.forward(image)
                # Backward Pass
                # Stochastic Gradient Descent
                dJdW2, dJdW1 = self.backwards(image, label, prediction, hidden)

            # Get training accuracy every 5 iterations
            if iteration % 5 == 0:
                predicted = self.predict(X)
                incorrect = np.count_nonzero(y - predicted)
                accuracy = (predicted.shape[0] - incorrect) / float(predicted.shape[0])
                print "Training Accuracy: %f" % accuracy
        return self.W1, self.W2

    def predict(self, X):
        # Propagate inputs forward through network
        if (self.W1 is not None) and (self.W2 is not None):
            yHat = []
            for image in X:
                z2 = np.dot(self.W1.T, image)
                a2 = np.append(self.tanh(z2), 1)
                z3 = np.dot(self.W2.T, a2)
                yHat.append(np.argmax(self.sigmoid(z3)))
            self.yHat = np.array(yHat)
            return self.yHat
        else:
            return None

    def forward(self, X):
        z2 = np.dot(self.W1.T, X)
        a2 = np.append(self.tanh(z2), 1)
        z3 = np.dot(self.W2.T, a2)
        yHat = self.sigmoid(z3)
        return a2, yHat

    def backwards(self, x, y, yHat, a2):
        delta3 = np.multiply(-(y - yHat), self.sigmoid_derivative(yHat))
        dJdW2 = np.outer(delta3, a2).T
        self.W2 -= (self.eta*dJdW2)
        delta2 = np.dot(self.W2, delta3) * self.tanh_derivative(a2)
        dJdW1 = np.outer(delta2[:-1], x).T
        self.W1 -= (self.eta*dJdW1)
        return dJdW2, dJdW1

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
        return 1 - np.square(self.tanh(x))

####################################################
DATA_DIR="/Users/David/dev/cs189/hw6/digit-dataset/"
train_data = scipy.io.loadmat(DATA_DIR+"train.mat")
test_data = scipy.io.loadmat(DATA_DIR+"test.mat")

# Load Data
X = train_data['train_images'].transpose().ravel().reshape((N_SAMPLES, IMG_SIZE)).astype(float)
Y = train_data['train_labels'].ravel()

# TODO: Pick preprocessing
xTrain = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
yTrain = Y
#X /= 256.0
import pdb; pdb.set_trace()

# Create Neural Network
net = NeuralNetwork()
w1, w2 = net.train(xTrain, yTrain, 1e-4, maxiter=10000)
np.save("w1", w1)
np.save("w2", w2)
#validate_labels = net.predict(xValidate)
pdb.set_trace()
#net.backwards(xTrain, yTrain)
