import scipy.io
import numpy as np
import math
import csv

from Node import *
from DTree import *
from Segmentor import *
from Impurity import *
from RForest import *


DATA_DIR="/Users/David/Documents/School/CS189/hw5/spam-dataset/"
data = scipy.io.loadmat(DATA_DIR+"spam_data")

VALIDATION_SIZE = 1000
TRAIN_SIZE = 5172

X = data['training_data']
Y = data['training_labels'].T.ravel()
randomIndex = np.random.choice(TRAIN_SIZE, TRAIN_SIZE, replace=False)

# Split Data
xTrain = X[randomIndex[:-VALIDATION_SIZE]]
yTrain = Y[randomIndex[:-VALIDATION_SIZE]]
xValidate = X[randomIndex[-VALIDATION_SIZE:]]
yValidate = Y[randomIndex[-VALIDATION_SIZE:]]
xTest = data['test_data']

segmentor = Segmentor()

print "============= Decision Tree =========="
tree = DTree(Impurity.impurity, segmentor)
tree.train(xTrain, yTrain)
labels = tree.predict(xValidate)

counts = np.bincount(tree.predict(xTrain) == yTrain)
error = 1.0 - (counts[True] / float(counts[True] + counts[False]))
print "Training Error: %f" % (error)

counts = np.bincount(labels == yValidate)
error = 1.0 - (counts[True] / float(counts[True] + counts[False]))
print "Validation Error: %f" % (error)

#import pdb; pdb.set_trace()

print "========== Random Forest =========="
forest = RForest(Impurity.impurity, segmentor, nTrees=30, randomness=10)
forest.train(xTrain, yTrain)
labels = forest.predict(xValidate)

counts = np.bincount(tree.predict(xTrain) == yTrain)
error = 1.0 - (counts[True] / float(counts[True] + counts[False]))
print "Training Error: %f" % (error)

counts = np.bincount(labels == yValidate)
error = 1.0 - (counts[True] / float(counts[True] + counts[False]))
print "Validation Error: %f" % (error)

#prediction = tree.predict(xTest)
#with open('results.csv', 'wb') as csvfile:
#    writer = csv.writer(csvfile)
#    writer.writerow(['Id', 'Category'])
#    for i in range(len(prediction)):
#        writer.writerow([i+1, prediction[i]])
