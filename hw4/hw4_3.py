import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math

import pdb



DATA_DIR="/Users/David/Documents/School/CS189/hw4/"
data = scipy.io.loadmat(DATA_DIR+"spam")

xTrain = np.mat(data['Xtrain'])
yTrain = np.mat(data['ytrain'])
xTest = np.mat(data['Xtest'])

# Preprocess Function: Takes 3 parameters; preprocess(matrix, matrixMEAN, matrixSTD)
preprocess1 = np.vectorize(lambda val, mean, std: (val - mean) / std)
preprocess2 = np.vectorize(lambda x: math.log(x + 0.1))
preprocess3 = np.vectorize(lambda x: x > 0)

# Preprocess data
# Mean 0 and Unit Variance
xTrainMean = xTrain.mean()
xTrainSTD = np.std(xTrain)
xTestMean = xTest.mean()
xTestSTD = np.std(xTest)

xTrain1 = preprocess1(xTrain, xTrainMean, xTrainSTD)
xTest1 = preprocess1(xTest, xTestMean, xTestSTD)
xTrain2 = preprocess2(xTrain)
xTest2 = preprocess2(xTest)
xTrain3 = preprocess3(xTrain)
xTest3 = preprocess3(xTest)

def calculate_mu(X, beta):
    mu = 1.0 / (1 + np.vectorize(math.exp)(-1 * X * beta ))
    return mu


def update_beta(X, Y, beta, L):
    mu = calculate_mu(X, beta)
    def diag(mu):
        ones = np.ones((len(mu), 1))
        retval = np.diagflat((ones - mu).T) * mu
        return np.diagflat(retval)
    A = diag(mu)
    identity = np.identity(len(beta))
    new_beta = beta - inv(2*L*identity - X.T * A * X) * (2*L*beta - X.T*(Y-mu))
    return new_beta

def batch_gradient(X, Y, step_mode, lam, tolerance, max_iterations):
    # Initialize beta to be zeros
    beta = np.mat(np.zeros(xTrain.shape[1])).T

pdb.set_trace()
