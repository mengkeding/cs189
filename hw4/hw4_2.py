import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import pdb

TRAINING_SIZE = 463715

data = np.loadtxt(open("YearPredictionMSD.txt","rb"), delimiter=",")

labels = np.mat(data[:,0]).T
values = np.mat(data[:,1:])


training_labels = labels[:TRAINING_SIZE]
training_values = values[:TRAINING_SIZE]

test_labels = labels[TRAINING_SIZE:]
test_values = values[TRAINING_SIZE:]

# Train beta
# X.T * X * beta = X.T Y
X = training_values.T * training_values
Y = training_values.T * training_labels
beta = linalg.solve(X, Y)


# Test
calc_labels = test_values * beta
RES = calc_labels - test_labels
# Matrix only has one value
RSS = np.asarray(0.5 * (RES.T * RES))[0][0]


print "RSS:" + str(RSS)
print "Max:" + str(max(calc_labels))
print "Min:" + str(min(calc_labels))

#pdb.set_trace()

#plt.stem(np.asarray(beta).T[0])
#plt.show()


