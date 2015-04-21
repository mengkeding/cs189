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

import pdb; pdb.set_trace()

