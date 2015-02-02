from sklearn import svm
from sklearn.metrics import confusion_matrix
import scipy.io
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import csv

import time
SPAM_DIR="/home/david/Documents/cs189/hw1/data/spam-dataset/"
SPAM_DATA = scipy.io.loadmat(SPAM_DIR+"spam_data")
training_data = SPAM_DATA['training_data'].astype(int)
training_labels = SPAM_DATA['training_labels'].astype(int)
test_data = SPAM_DATA['test_data'].astype(int)

N_SAMPLES = 5172
VALIDATION_SIZE = 1000

# Trains and Validates Spam Classifier: Problem 4
def trainValidate(n, c=1.0):
    clf = svm.SVC(C=c, kernel='linear')
    TRAIN_SIZE = n
    indices = random.sample(xrange(len(training_data)), TRAIN_SIZE)
    x = []
    y = []
    for i in indices:
        x.append(training_data[i])
        y.append(training_labels[0][i])
    x = np.array(x)
    y = np.array(y).ravel()
    clf.fit(x, y)
    val_indices = random.sample(xrange(len(training_data)), VALIDATION_SIZE)
    val_x = []
    val_y = []
    for i in val_indices:
        val_x.append(training_data[i])
        val_y.append(training_labels[0][i])
    val_x = np.array(val_x)
    val_y = np.array(val_y).ravel()
    accuracy = clf.score(val_x, val_y)
    return accuracy

# k-fold validation: k=10; Problem 4
def tenFold(c=1.0):
    PARTITION_SIZE = len(training_data) / 9
    indices = [ x for x in range(PARTITION_SIZE, N_SAMPLES, PARTITION_SIZE) ]

    data = np.split(training_data, indices)
    labels = np.split(training_labels.T, indices)
    values = []

    clf = svm.SVC(C=c, kernel='linear')

    for i in range(len(data)):
        print "case "+str(i)
        val_x = data[i]
        val_y = labels[i].ravel()
        x = np.concatenate(data[0:i]+data[i+1:])
        y = np.concatenate(labels[0:i]+labels[i+1:]).ravel()

        clf.fit(x, y)
        accuracy = clf.score(val_x, val_y)
        values.append(accuracy)

    average = sum(values) / len(values)
    return average

# Tests data for kaggle and writes to csv
def test(n, c=1.0):
    clf = svm.SVC(C=c, kernel='linear')
    TRAIN_SIZE = n
    index = random.sample(xrange(len(training_data)), TRAIN_SIZE)
    x = []
    y = []
    for i in index:
        x.append(training_data[i])
        y.append(training_labels[0][i])
    x = np.array(x)
    y = np.array(y)
    clf.fit(x, y)
    prediction = clf.predict(test_data)
    with open('spam.csv' ,'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        for i in range(len(prediction)):
            writer.writerow([i+1, prediction[i]])


#training_sizes = [ 10, 50, 100, 1000, 2000, 5000]
#for n in training_sizes:
#    print "Training Size: "+str(n)+" Accuracy: "+str(trainValidate(n)*100)+"%"

#c_values = [ pow(10, x) for x in range(-5,6) ]
#averages = []
#for c in c_values:
#    a = tenFold(c)
#    print "C value: "+str(c)+" Accuracy: "+str(a*100)+"%"
#    averages.append(a)
#
#index, value = max(enumerate(averages), key=operator.itemgetter(1))
#print "Best C: "+str(c_values[index])
#print "Best Average: "+str(value)

test(N_SAMPLES, 10)
