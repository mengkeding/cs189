import numpy as np
from numpy.linalg import inv
import math

import pdb

# X and Beta are numpy matrices
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


X = np.mat( [[0, 3],
            [1, 3],
            [0, 1],
            [1, 1]
        ])
# Append 1 to x_i vectors
ones = np.ones(len(X))
X = np.hstack([X, np.mat(ones).T])

Y = np.mat( [[1],
            [1],
            [0],
            [0]
        ])
lam = 0.07
beta_0 = np.mat([-2, 1, 0]).T

# Problem 1
mu_0 = calculate_mu(X, beta_0)
print "mu_0:"
print mu_0

# Problem 2
beta_1 = update_beta(X, Y, beta_0, lam)
print "beta_1:"
print beta_1

# Problem 3
mu_1 = calculate_mu(X, beta_1)
print "mu_1:"
print mu_1

# Problem 4
beta_2 = update_beta(X, Y, beta_1, lam)
print "beta_2:"
print beta_2
