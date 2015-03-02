import scipy.io
import numpy as np
import random
import csv
from collections import defaultdict
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


# Variables
# Mac Directory
DATA_DIR="/Users/David/Documents/School/CS189/HW3/data/digit-dataset/"
# Linux Directory
#DATA_DIR="/home/david/Documents/CS189/HW3/data/digit-dataset/"

IMG_SIZE = 28*28 # known image size
VALIDATION_SIZE = 10000
N_SAMPLES = 60000
TEST_SIZE = 5000
TRAIN_SIZE = 60000

# Load Training Set
train = scipy.io.loadmat(DATA_DIR+"train")
train_labels = train['train_label']
train_images = train['train_image'].transpose([2,0,1])
test_images = scipy.io.loadmat(DATA_DIR+"test")['test_image'].transpose([2,0,1]).ravel().reshape((TEST_SIZE, IMG_SIZE))
test_labels = scipy.io.loadmat(DATA_DIR+"test")['test_label']

# Flatten Training Set
flat = train_images.ravel().reshape((N_SAMPLES, IMG_SIZE))

# Partition Training Set
index = random.sample(xrange(len(flat)), TRAIN_SIZE)
X = []
y = []
for i in index:
    X.append(flat[i])
    y.append(train_labels[i][0])
X = np.array(X)
y = np.array(y).reshape(len(y), 1)

# Bucket the images into labels
images = defaultdict(list)
for img, label in zip(X, y):
    img = img.astype("uint64")
    # Normalize
    norm = np.linalg.norm(img)
    images[label[0]].append(img / norm)

covariances = defaultdict(int)
means = defaultdict(int)
# Get means and covariances matricies for each class
alpha = 1e-5
for label in images:
    v = np.array(images[label])
    cov = np.cov(v.T)
    mean = np.mean(v, axis=0)
    covariances[label] = cov + alpha * np.identity(784)
    means[label] = mean

priors = defaultdict(int)
for label in images:
    priors[label] = len(images[label]) / float(N_SAMPLES)

#for label in covariances:
#    cov = covariances[label]
#    plt.imshow(cov)
#    plt.colorbar()
#    plt.title("Class Label: "+str(label))
#    plt.show()

#alpha = 1e-5 * np.identity(784)
s_overall = np.mean(covariances.values(), axis=0) #+ alpha

gaussians = []
for label in means:
    gaussians.append((multivariate_normal(mean=means[label], cov=s_overall), label))

count = 0
for true_label, img in zip(test_labels, test_images):
    img = img.astype('uint64')
    img = img / np.sqrt(img.dot(img))
    _, label = max(gaussians, key=lambda gaussian: gaussian[0].logpdf(img))
    if label == true_label:
        count += 1
accuracy = count / float(len(test_images))
print "Accuracy: " + str(accuracy*100) + "%"

kaggle_images = scipy.io.loadmat(DATA_DIR+"kaggle")['kaggle_image'].transpose([2,0,1]).ravel().reshape((TEST_SIZE, IMG_SIZE))
#import pdb; pdb.set_trace()

with open('results_digits.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Category'])
    count = 1
    for img in kaggle_images:
        img = img.astype('uint64')
        img = img / np.sqrt(img.dot(img))
        _, label = max(gaussians, key=lambda gaussian: gaussian[0].logpdf(img))
        writer.writerow([count, label])
        count += 1
print "Done"

