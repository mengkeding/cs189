import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
# WARM-UP

def average(data):
    means = np.mean(data, axis=0)
    return means

average_scores = average(train_data)

average_validation = validation_features
average_labels = []

for index in range(len(average_validation)):
    joke = average_validation[index][1]
    if average_scores[joke-1] > 0:
        label = 1
    else:
        label = 0
    average_labels.append([label])
average_labels = np.array(average_labels)

error = np.count_nonzero(validation_labels - average_labels) / float(len(validation_labels))

#################################################################################

pdb.set_trace()

