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
accuracy = 1.0 - error

print "==========AVERAGE RATING=========="
print "Validation Accuracy: %f" % (accuracy)

def euclidian_distance(user1, user2):
    return np.linalg.norm(user1 - user2)

# TODO: optimize
def kmeans_recommendation(k):
    recommendation_matrix = []
    for index in xrange(N_USERS):
        if index % 100 == 0:
            print "Index %d / %d" % (index, N_USERS)
        neighbors = find_nearest_neighbors(index, k)
        vector = np.zeros(train_data[index].shape)
        while not neighbors.empty():
            neighbor_index = neighbors.get()[1]
            vector = np.add(vector, train_data[neighbor_index])
        vector = np.divide(vector, float(k))
        recommendation_matrix.append(vector)
    recommendation_matrix = np.array(vector)
    return recommendation_matrix




def find_nearest_neighbors(user_index, k):
    user = train_data[user_index]
    best = Queue.PriorityQueue(k)
    for index in xrange(N_USERS):
        # Skip self
        if index == user_index:
            continue
        # Get negative distance between neighbors
        distance = -euclidian_distance(user, train_data[index])
        # if under k neighbors found, append to best
        if not best.full():
            best.put((distance, index))
        # check if you beat any
        else:
            worst = best.get()
            if worst[0] < distance:
                best.put((distance, index))
            else:
                best.put(worst)
    return best

#################################################################################
#test = find_nearest_neighbors(0, 10)
test = kmeans_recommendation(10)
pdb.set_trace()

