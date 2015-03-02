import scipy.io
import numpy as np
import random
import csv
from collections import defaultdict
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


# Variables
# Mac Directory
DATA_DIR="/Users/David/Documents/School/CS189/HW3/data/spam-dataset/"
# Linux Directory
#DATA_DIR="/home/david/Documents/CS189/HW3/data/spam-dataset/"

SPAM_DATA = scipy.io.loadmat(DATA_DIR+"spam_data")
training_data = SPAM_DATA['training_data'].astype(int)
training_labels = SPAM_DATA['training_labels'].astype(int)
test_data = SPAM_DATA['test_data'].astype(int)

N_SAMPLES = 5172
TRAIN_SIZE = 4000
VALIDATION_SIZE = 1000

# Partition Training Set
index = random.sample(xrange(len(training_data)), TRAIN_SIZE)
X = []
y = []
for i in index:
    X.append(training_data[i])
    y.append(training_labels[0][i])
X = np.array(X)
y = np.array(y).ravel()
val_indices = random.sample(xrange(len(training_data)), VALIDATION_SIZE)
val_x = []
val_y = []
for i in val_indices:
    val_x.append(training_data[i])
    val_y.append(training_labels[0][i])
val_x = np.array(val_x)
val_y = np.array(val_y).ravel()

# Buckets
emails = defaultdict(list)
for email, label in zip(X, y):
    email = email.astype("uint64")
    # Normalize
    norm = np.linalg.norm(email) + 1e-5
    emails[label].append(email / norm)

covariances = defaultdict(int)
means = defaultdict(int)
# Get means and covariances matricies for each class
alpha = 1e-5
for label in emails:
    v = np.array(emails[label])
    cov = np.cov(v.T)
    mean = np.mean(v, axis=0)
    covariances[label] = cov + alpha * np.identity(32)
    means[label] = mean

s_overall = np.mean(covariances.values(), axis=0)

gaussians = []
for label in means:
    gaussians.append((multivariate_normal(mean=means[label], cov=s_overall), label))

count = 0
for true_label, email in zip(val_y, val_x):
    email = email.astype('uint64')
    email = email / (np.sqrt(email.dot(email)) + alpha)
    _ , label = max(gaussians, key=lambda gaussian: gaussian[0].logpdf(email))
    if label == true_label:
        count += 1
accuracy = count / float(len(val_y))
print "Accuracy: " + str(accuracy*100) + "%"

with open('results_spam.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Category'])
    count = 1
    for email in test_data:
        email = email.astype('uint64')
        email = email / (np.sqrt(email.dot(email)) + alpha)
        _, label = max(gaussians, key=lambda gaussian: gaussian[0].logpdf(email))
        writer.writerow([count, label])
        count += 1
print "Done"
