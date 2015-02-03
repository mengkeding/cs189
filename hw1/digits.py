from sklearn import svm
from sklearn.metrics import confusion_matrix
import scipy.io
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import csv

import time

# Variables
DIGIT_DIR="/home/david/Documents/cs189/hw1/data/digit-dataset/"
IMG_SIZE = 28*28 # known image size
VALIDATION_SIZE = 10000
N_SAMPLES = 60000
TEST_SIZE = 10000

# Load Training Set
train = scipy.io.loadmat(DIGIT_DIR+"train")
train_labels = train['train_labels']
train_images = train['train_images'].transpose([2,0,1])
test_images = scipy.io.loadmat(DIGIT_DIR+"test")['test_images'].transpose([2,0,1]).ravel().reshape((TEST_SIZE, IMG_SIZE))

# Flatten Training Set
flat = train_images.ravel().reshape((N_SAMPLES, IMG_SIZE))


# Problem 1 & 2
def trainValidateSVM(n, cv=1.0):
    clf = svm.SVC(C=cv, kernel='linear')
    #print clf
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

    #print "training on "+str(TRAIN_SIZE)+" samples"
    start = time.time()
    clf.fit(X, y)
    end = time.time()
    #print "done: "+str(end - start)
    #print "validating on "+str(VALIDATION_SIZE)+" samples"
    validation_indices = random.sample(xrange(len(flat)), VALIDATION_SIZE)
    x_val = []
    y_val = []
    for i in validation_indices:
        x_val.append(flat[i])
        y_val.append(train_labels[i][0])
    predict = clf.predict(x_val)
    cm = confusion_matrix(y_val, predict)

    # Calculate accuracy
    accuracy = clf.score(x_val, y_val)
    #print "done; printing confusion matrix:"
    # print confusion_matrix
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.draw()
    return accuracy

# Problem 3
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
    print clf
    values = []
    for i in range(len(images)):
        print "case "+str(i)
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
        #prediction = clf.predict(validation_images)
        accuracy = clf.score(validation_images, validation_labels)
        values.append(accuracy)
    print values
    average = sum(values) / len(values)
    return average

# Problem 3
def test(n, c=1.0):
    clf = svm.SVC(kernel='linear', C=c)
    TRAIN_SIZE = n
    index = random.sample(xrange(len(flat)), TRAIN_SIZE)
    X = []
    y = []
    for i in index:
        X.append(flat[i])
        y.append(train_labels[i][0])
    X = np.array(X)
    y = np.array(y)
    clf.fit(X, y)
    prediction = clf.predict(test_images)
    print prediction
    print len(prediction)
    with open('results.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        for i in range(len(prediction)):
            writer.writerow([i+1, prediction[i]])



#sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
#for n in sample_sizes:
#    print "Score for "+str(n)+": "+str(trainValidateSVM(n)*100)+"%"

#c_values = [ pow(10, x) for x in range(-5,6) ]
#for c in c_values:
#    print "C value: "+str(c)+" Accuracy: "+str(trainValidateSVM(10000, c)*100)+"%"

#print "Average 10-fold: "+str(tenFold(10)*100)+"%"

#c_values = [ pow(10, x) for x in range(-5,6) ]
#averages = []
#for c in c_values:
#    print "Testing C: "+str(c)
#    averages.append(tenFold(c))
#print c_values
#print averages
#index, value = max(enumerate(averages), key=operator.itemgetter(1))
#print "Best C: "+str(c_values[index])
#print "Best Average: "+str(value)

print str(trainValidateSVM(10000, 1e-05)*100)+"%"
#start = time.time()
#test(30000, 1e-05)
#end = time.time()
#print str(end - start)
#
plt.show()
