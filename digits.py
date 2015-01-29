from sklearn import svm
from sklearn.metrics import confusion_matrix
import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt

import time

# Variables
#Mac
DIGIT_DIR="/Users/David/Documents/School/CS189/hw1/data/digit-dataset/"
#Linux
#DIGIT_DIR="/home/david/Documents/cs189/hw1/data/digit-dataset/"
IMG_SIZE = 28*28 # known image size
VALIDATION_SIZE = 10000
N_SAMPLES = 60000

# Load Training Set
train = scipy.io.loadmat(DIGIT_DIR+"train")
train_labels = train['train_labels']
train_images = train['train_images'].transpose([2,0,1])

# Flatten Training Set
flat = train_images.ravel().reshape((N_SAMPLES, IMG_SIZE))


def trainValidateSVM(n, c=1.0):
    clf = svm.SVC(kernel='linear')
    TRAIN_SIZE = n

    # Partition Training Set
    index = random.sample(xrange(len(flat)), TRAIN_SIZE)
    X = []
    y = []
    for i in index:
        X.append(flat[i])
        y.append(train_labels[i][0])
    X = np.array(X)
    y = np.array(y)

    print "training on "+str(TRAIN_SIZE)+" samples"
    start = time.time()
    clf.fit(X, y)
    end = time.time()
    print "done: "+str(end - start)

    # Initialize Confusion Matrix
    #confusion_matrix = np.zeros((10,10), dtype=np.int)

    print "validating on "+str(VALIDATION_SIZE)+" samples"
    validation_indices = random.sample(xrange(len(flat)), VALIDATION_SIZE)
    # Add to confusion matrix while counting
    count = 0
    validate_x = []
    validate_y = []
    for i in validation_indices:
        validate_x.append(flat[i])
        validate_y.append(train_labels[i][0])
    predict = clf.predict(validate_x)
    cm = confusion_matrix(validate_y, predict)

    # Calculate accuracy
    accuracy = clf.score(validate_x, validate_y)
    print "done; printing confusion matrix:"
    #print confusion_matrix
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return accuracy

# C value for 'Penalty parameter C of the error term' in SVC
# Defaults to 1.0
def tenFold(c=1.0):
    # Sample from training set
    index = random.sample(xrange(len(flat)), VALIDATION_SIZE)
    X = []
    y = []
    for i in index:
        X.append(flat[i])
        y.append(train_labels[i][0])
    images = np.array(X)
    labels = np.array(y)

    indices = [ x for x in range(VALIDATION_SIZE / 10, VALIDATION_SIZE, VALIDATION_SIZE / 10) ]
    images = np.split(images, indices)
    labels = np.split(labels, indices)
    clf = svm.SVC(kernel='linear', C=c)
    values = []
    for i in range(len(images)-1):
        print "case "+str(i)
        count = 0
        validation_images = images[i]
        validation_labels = labels[i]
        training_images = np.concatenate(images[0:i]+images[i+1:])
        training_labels = np.concatenate(labels[0:i]+labels[i+1:])
        print "training"
        start = time.time()
        clf.fit(training_images, training_labels)
        end = time.time()
        print "done: "+str(end - start)
        print "validation"
        for i in range(len(validation_images)-1):
            prediction = clf.predict(validation_images[i])
            label = validation_labels[i]
            if prediction == label:
                count += 1
        values.append(float(count) / len(validation_images))
    print values
    average = sum(values) / len(values)
    return average


#sample_sizes = [100, 200, 500, 1000, 5000, 10000]
#for n in sample_sizes:
#    print "Score for "+str(n)+": "+str(trainValidateSVM(n)*100)+"%"
print str(trainValidateSVM(10000)*100)+"%"

#print "Average 10-fold: "+str(tenFold()*100)+"%"
#c_values = [ pow(10, x) for x in range(-5,6) ]
#d = {}
#for c in c_values:
#    print "Testing C: "+str(c)
#    d[c] = tenFold(c)
#
#print d
