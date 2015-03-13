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
BATCH_MAX_ITERATIONS=1000
STO_MAX_ITERATIONS=5000

preprocess1 = lambda mat: (mat - mat.mean(0)) / mat.std(0)
preprocess2 = np.vectorize(lambda x: math.log(x + 0.1))
preprocess3 = np.vectorize(lambda x: int(x > 0))

# Preprocess data
# Mean 0 and Unit Variance
xTrain1 = preprocess1(xTrain)
xTest1 = preprocess1(xTest)
xTrain2 = preprocess2(xTrain)
xTest2 = preprocess2(xTest)
xTrain3 = preprocess3(xTrain)
xTest3 = preprocess3(xTest)

def calculate_mu(X, beta):
    mu = 1.0 / (1.0 + np.vectorize(math.exp)(-1.0 * X * beta ))
    return mu

def update_beta(X, Y, beta, L, eta, stochastic=False):
    if not stochastic:
        mu = calculate_mu(X, beta)
        new_beta = beta - eta * (2*L*beta - X.T * (Y - mu))
    else:
        i = np.random.randint(0, len(X))
        Xi = X[i]
        Yi = Y[i]
        mu = calculate_mu(Xi, beta)
        new_beta = beta - eta * (2*L*beta - Xi.T * (Yi - mu))
    return new_beta

def training_loss(X, Y, beta):
    total_loss = 0.0
    for i in range(len(X)):
        mu = calculate_mu(X[i], beta)
        total_loss += -1.0 * (Y[i] * math.log(mu) + (1 - Y[i]) * math.log(1-mu))
    return float(total_loss)

def batch_gradient(X, Y):
    L = 1e-6
    eta = 1e-5
    # Initialize beta to be zeros
    beta = np.mat(np.zeros(xTrain.shape[1])).T
    change = float("inf")
    losses = []
    for i in range(BATCH_MAX_ITERATIONS):
        prev = beta
        beta = update_beta(X, Y, beta, L, eta)
        change = np.linalg.norm(prev - beta)
        if i % 100 == 0:
            losses.append(training_loss(X, Y, beta))
        if change < 1e-6:
            break
    plt.plot(list(range(0, len(losses)*10, 10)), losses)
    plt.show()
    return beta

def stochastic_gradient(X, Y, variable=False):
    L = 1e-6
    eta = 1e-5
    constant_eta = 1e-5
    # Initialize beta to be zeros
    beta = np.mat(np.zeros(xTrain.shape[1])).T
    change = float("inf")
    losses = []
    for i in range(STO_MAX_ITERATIONS):
        if variable:
            eta = constant_eta / float(i+1)
        prev = beta
        beta = update_beta(X, Y, beta, L, eta, stochastic=True)
        change = np.linalg.norm(prev - beta)
        if i % 100 == 0:
            losses.append(training_loss(X, Y, beta))
        if change < 1e-10:
            break
    plt.plot(list(range(0, len(losses)*100, 100)), losses)
    plt.show()
    return beta

batch_beta1 = batch_gradient(xTrain1, yTrain)
batch_beta2 = batch_gradient(xTrain2, yTrain)
batch_beta3 = batch_gradient(xTrain3, yTrain)

sto_beta1 = stochastic_gradient(xTrain1, yTrain)
sto_beta2 = stochastic_gradient(xTrain2, yTrain)
sto_beta3 = stochastic_gradient(xTrain3, yTrain)

varLearning_beta1 = stochastic_gradient(xTrain1, yTrain, variable=True)
varLearning_beta2 = stochastic_gradient(xTrain2, yTrain, variable=True)
varLearning_beta3 = stochastic_gradient(xTrain3, yTrain, variable=True)
