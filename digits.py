from sklearn import svm
import scipy.io
import numpy as np
import random

import time

# Variables
#Mac
#DIGIT_DIR="/Users/David/Documents/School/CS189/hw1/data/digit-dataset/"
#Linux
DIGIT_DIR="/home/david/Documents/cs189/hw1/data/digit-dataset/"
IMG_SIZE = 28*28 # known image size
TRAIN_SIZE = 100
VALIDATION_SIZE = 10000
N_SAMPLES = 60000
clf = svm.SVC(kernel='linear')

# Load Training Set
train = scipy.io.loadmat(DIGIT_DIR+"train")
train_labels = train['train_labels']
train_images = train['train_images']

# Flatten Training Set
flat = train_images.transpose([2,0,1]).ravel().reshape((N_SAMPLES, IMG_SIZE))
#flat = np.array([ np.array(flat[x:x+IMG_SIZE]) for x in range(0, len(flat), IMG_SIZE) ])

def trainSVM(n):
    global TRAIN_SIZE
    global clf
    TRAIN_SIZE = n

    # Partition Training Set
    index = random.sample(xrange(len(flat)), TRAIN_SIZE)
    X = []
    y = []
    for i in index:
        X.append(flat[i])
        y.append(train_labels[i][0])

    print "training on "+str(TRAIN_SIZE)+" samples"
    start = time.time()
    clf.fit(X, y)
    end = time.time()
    print "done: "+str(end - start)
    print clf

def validateSVM():
    validation_indices = random.sample(xrange(len(flat)), VALIDATION_SIZE)
    count = 0
    for i in validation_indices:
        prediction = clf.predict(flat[i])
        label = train_labels[i]
        if prediction == label:
            count = count + 1
    accuracy = float(count) / VALIDATION_SIZE
    return accuracy

trainSVM(10000)
print str(validateSVM()*100)+"%"
