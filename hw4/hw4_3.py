import scipy.io
import numpy as np
import matplotlib.pyplot as plt

import pdb



DATA_DIR="/Users/David/Documents/School/CS189/hw4/"
data = scipy.io.loadmat(DATA_DIR+"spam")

xTrain = np.mat(data['Xtrain'])
yTrain = np.mat(data['ytrain'])
xTest = np.mat(data['Xtest'])

# Preprocess data
# Mean 0 and Unit Variance
preprocess = np.vectorize(lambda val, mean: val - mean)

# Check Mean of columns and variance

pdb.set_trace()
