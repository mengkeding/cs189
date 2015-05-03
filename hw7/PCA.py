import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Queue
import csv

import pdb

#################################################################################
# CLEAN UP / PREPROCESSING
M_JOKES = 100
N_USERS = 24983

DATA_DIR='/Users/David/dev/cs189/hw7/joke_data/'
data = scipy.io.loadmat(DATA_DIR+'joke_train.mat')

train_data = np.nan_to_num(data['train'])

validation = np.loadtxt(DATA_DIR+'validation.txt', dtype=int, delimiter=',')
validation_features = np.delete(validation, np.s_[2:], 1)
validation_labels = np.delete(validation, np.s_[:2], 1)

#################################################################################

class PCA():
    def __init__(self, mat):
        pass

pdb.set_trace()
