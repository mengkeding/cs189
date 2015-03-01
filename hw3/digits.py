import time
import scipy.io
import numpy as np
import random
import operator
import csv
from collections import defaultdict


# Variables
# Mac Directory
DATA_DIR="/Users/David/Documents/School/CS189/HW3/data/digit-dataset/"
# Linux Directory
#DATA_DIR="/home/david/Documents/CS189/HW3/data/digit-dataset/"

IMG_SIZE = 28*28 # known image size
VALIDATION_SIZE = 10000
N_SAMPLES = 60000
TEST_SIZE = 5000

# Load Training Set
train = scipy.io.loadmat(DATA_DIR+"train")
train_labels = train['train_label']
train_images = train['train_image'].transpose([2,0,1])
test_images = scipy.io.loadmat(DATA_DIR+"test")['test_image'].transpose([2,0,1]).ravel().reshape((TEST_SIZE, IMG_SIZE))

# Flatten Training Set
flat = train_images.ravel().reshape((N_SAMPLES, IMG_SIZE))

# Bucket the images into labels
images = defaultdict(list)
for img, label in zip(flat, train_labels):
    img = img.astype("uint64")
    # Normalize
    norm = np.linalg.norm(img)
    images[label[0]].append(img / norm)
    import pdb; pdb.set_trace()

covariances = defaultdict(int)
means = defaultdict(int)
# Get means and covariances matricies for each class
for label in images:
    v = np.array(images[label])
    cov = np.cov(v.T)
    mean = np.mean(v, axis=0)
    covariances[label] = cov
    means[label] = mean



