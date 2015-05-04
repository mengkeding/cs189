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

print "==========AVERAGE RATING=========="
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

print "Validation Accuracy: %f" % (accuracy)

def euclidian_distance(user1, user2):
    return np.linalg.norm(user1 - user2)

# DO NOT USE
# Calculates kneighbors for all users
# Incredibly slow
def kneighbors_recommendation_full(k):
    neighbors = []
    recommendations = []
    for i in xrange(N_USERS):
        user1 = train_data[i]
        user_distance = []
        if i % 100 == 0:
            print "Index %d / %d" % (i, N_USERS)
        for j in xrange(N_USERS):
            if j == i:
                continue
            user2 = train_data[j]
            distance = euclidian_distance(user1, user2)
            user_distance.append((j, distance))
        neighbors.append(sorted(user_distance, key=lambda tup: tup[1])[:k])
    for i in xrange(N_USERS):
        vector = np.zeros(train_data[0].shape)
        for user, distance in neighbors[i]:
            vector = np.add(vector, train_data[user])
        vector = np.divide(vector, float(k))
        recommendations.append(vector)
    return np.array(recommendations)

# Only compute neighbors for those in validation set
def kneighbors_recommendation(k):
    recommendations = {}
    for index in range(len(validation_features)):
        if index % 100 == 0:
            print "Index %d / %d" % (index, len(validation_features))
        user = validation_features[index][0]
        user1 = train_data[user-1]
        neighbors = []
        # if computed, don't recompute
        if user in recommendations.keys():
            continue
        else:
            # Need to check all users
            for i in xrange(N_USERS):
                if user-1 == i:
                    continue
                user2 = train_data[i]
                distance = euclidian_distance(user1, user2)
                # Append index, distance
                neighbors.append((i, distance))
            # Sort by distance
            neighbors = sorted(neighbors, key=lambda tup: tup[1])[:k]
            vector = np.zeros(train_data[0].shape)
            for i, d in neighbors:
                vector = np.add(vector, train_data[i])
            vector = np.divide(vector, float(k))
            recommendations[user] = vector
    return recommendations

print "==========K_NEIGHBORS AVERAGE RATING=========="
k_neighbors = kneighbors_recommendation(10)
neighbors_labels = []
for index in range(len(validation_features)):
    user, joke = average_validation[index]
    if k_neighbors[user][joke-1] > 0:
        label = 1
    else:
        label = 0
    neighbors_labels.append([label])
neighbors_labels = np.array(neighbors_labels)

error = np.count_nonzero(validation_labels - neighbors_labels) / float(len(validation_labels))
accuracy = 1.0 - error
print "Validation Accuracy: %f" % (accuracy)
#################################################################################
#test = find_nearest_neighbors(0, 10)
pdb.set_trace()

