from sklearn import svm
import scipy.io
import numpy as np
import random

DIGIT_DIR="/Users/David/Documents/School/CS189/hw1/data/digit-dataset/"
IMG_SIZE = 28*28 # known training size
TRAIN_SIZE = 100
N_SAMPLES = 60000 # Default to all samples
clf = svm.SVC()

train = scipy.io.loadmat(DIGIT_DIR+"train")

train_labels = train['train_labels']
train_images = train['train_images']

def trainSVM(n):
    global TRAIN_SIZE
    global clf
    TRAIN_SIZE = n
    tmp = train_images.transpose([2,0,1])
    flat = tmp.ravel()
    flat = [ flat[x:x+IMG_SIZE] for x in range(0, len(flat), IMG_SIZE) ]

    # Random indices
    index = random.sample(xrange(len(flat)), TRAIN_SIZE)
    X = []
    y = []

    for i in index:
        X.append(flat[i])
        y.append(train_labels[i][0])

    print "training on "+str(TRAIN_SIZE)+" samples"
    clf.fit(X, y)
    print "done"
    print clf

trainSVM(10)
