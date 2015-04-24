import scipy.io
import numpy as np
import csv

import pdb

IMG_SIZE = 28*28
N_SAMPLES = 60000
TEST_SIZE = 10000
VALIDATION_SIZE = 50000

class NeuralNetwork():
    # Initialize Neural Network
    def __init__(self, entropy=False):
        # Layer Sizes
        self.input_layer_size = IMG_SIZE
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
    def train(self, X, y, eta, maxiter=1000):
        # Weights and bias
        UPPER_BOUND = 1.0
        LOWER_BOUND = -1.0
        self.W1 = (UPPER_BOUND - LOWER_BOUND) * np.random.randn(self.input_layer_size+1, self.hidden_layer_size) + LOWER_BOUND
        self.W2 = (UPPER_BOUND - LOWER_BOUND) * np.random.randn(self.hidden_layer_size+1, self.output_layer_size) + LOWER_BOUND

        #self.W1 = np.random.normal(loc=0, scale=0.01, size=(self.input_layer_size+1, self.hidden_layer_size))
        #self.W2 = np.random.normal(loc=0, scale=0.01, size=(self.hidden_layer_size+1, self.output_layer_size))


        # Learning Rate
        self.eta = eta

        # Pad for bias
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        # Transform Y
        def transform_y(digit):
            vector = np.zeros(10)
            vector[digit] = 1
            return vector
        num = 0

        for iteration in range(maxiter):

            # Random index to split training set for validation
            self.randomIndex = np.random.choice(X.shape[0], X.shape[0], replace=True)

            for i in self.randomIndex:
                ## Select a data point
                image = X[i]
                label = transform_y(y[i])
                # Forward Pass
                hidden, prediction = self.forward(image)
                # Backward Pass & Stochastic Gradient Descent
                dJdW2, dJdW1 = self.backwards(image, label, prediction, hidden)

                # Get training accuracy every 1000 iterations
                if num % 1000 == 0 or num == maxiter-1:
                    predicted = self.predict(X)
                    incorrect = np.count_nonzero(y - predicted)
                    accuracy = (predicted.shape[0] - incorrect) / float(predicted.shape[0])
                    np.save("w1", self.W1)
                    np.save("w2", self.W2)
                    print "Iteration: %d" % (num)
                    print "Training Accuracy: %f" % (accuracy)
                    if accuracy > 0.95:
                        np.save("w1", self.W1)
                        np.save("w2", self.W2)
                        return self.W1, self.W2
                num += 1
            if num > maxiter:
                break
        return self.W1, self.W2

    def predict(self, X, pad=False):
        # Propagate inputs forward through network
        if pad:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
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
        if self.entropy == True:
            # Cross Entropy
            # Stochastic Gradient Descent: Output to Hidden
            delta3 = yHat - y
            dJdW2 = np.outer(delta3, a2).T
            # Update W2
            self.W2 -= (self.eta*dJdW2)
            # Stochastic Gradient Descent: Hidden to Input
            delta2 = np.dot(self.W2, delta3) * self.tanh_derivative(a2)
            dJdW1 = np.outer(delta2[:-1], x).T
            # Update W1
            self.W1 -= (self.eta*dJdW1)
        else:
            # Mean squared
            # Stochastic Gradient Descent: Output to Hidden
            delta3 = np.multiply(-(y - yHat), self.sigmoid_derivative(yHat))
            dJdW2 = np.outer(delta3, a2).T

            # Update W2
            self.W2 -= (self.eta*dJdW2)
            # Stochastic Gradient Descent: Hidden to Input
            delta2 = np.dot(self.W2, delta3) * self.tanh_derivative(a2)
            dJdW1 = np.outer(delta2[:-1], x).T

            # Update W1
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
        return J

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def sigmoid_derivative(self, x):
        return np.multiply(self.sigmoid(x), (1.0 - self.sigmoid(x)))

    def tanh_derivative(self, x):
        return 1.0 - np.square(self.tanh(x))

####################################################
DATA_DIR="/Users/David/dev/cs189/hw6/digit-dataset/"
train_data = scipy.io.loadmat(DATA_DIR+"train.mat")
test_data = scipy.io.loadmat(DATA_DIR+"test.mat")

# Load Data
X = train_data['train_images'].transpose().ravel().reshape((N_SAMPLES, IMG_SIZE)).astype(float)
Y = train_data['train_labels'].ravel()

# Preprocess
# Center and normalize
xTrain = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
yTrain = Y

# Straight division
#xTrain = X / 256.0
#yTrain = Y

# Create Neural Network
net = NeuralNetwork()
#net = NeuralNetwork(entropy=True)
w1, w2 = net.train(xTrain, yTrain, 0.1, maxiter=700000)
np.save("w1", w1)
np.save("w2", w2)
#validate_labels = net.predict(xValidate)
net.W1 = np.load("w1.npy")
net.W2 = np.load("w2.npy")

predicted = net.predict(xTrain, pad=True)
incorrect = np.count_nonzero(yTrain - predicted)
accuracy = (predicted.shape[0] - incorrect) / float(predicted.shape[0])
print "Training Accuracy: %f" % accuracy
pdb.set_trace()


test = test_data['test_images'].transpose().ravel().reshape((10000, IMG_SIZE)).astype(float)
xTest = (test - np.mean(test, axis=0)) / (np.std(test, axis=0) + 1e-9)
yTest = net.predict(xTest, pad=True)

with open('results.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Category'])
    for i in range(len(yTest)):
        writer.writerow([i+1, yTest[i]])

pdb.set_trace()
