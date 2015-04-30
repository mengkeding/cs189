import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pdb

#################################################################################

IMG_SIZE = 28*28
N_SAMPLES = 60000

DATA_DIR='/Users/David/dev/cs189/hw7/mnist_data/'
data = scipy.io.loadmat(DATA_DIR+'images.mat')

images = data['images'].transpose([2, 0, 1]).reshape((N_SAMPLES, IMG_SIZE)).astype(float)

class KMeans():
    # Initialize K Means
    def __init__(self, data, k):
        self.data = data
        self.k = k
        # Initialize Centroids
        self.centroids = [data[i] for i in np.random.choice(data.shape[0], k, replace=False)]
        # Pick initial clusters
        # self.clusters should never be None after initialization
        self.clusters = None
        self.pick_all_clusters()
        # Calculate initial loss
        self.loss = self.objective_function()
        self.losses = []
        for iteration in range(100):
            print "Iteration %d" % (iteration)
            self.compute_means()
            print "Recalculated Means"
            self.pick_all_clusters()
            print "Repicked clusters"
            tmp = self.objective_function()
            if tmp < self.loss:
                print "===========BETTER============"
                pass
            elif tmp == self.loss:
                print "Converged"
                self.loss = tmp
                break
            else:
                print "===========WORSE============"
            difference = abs(self.loss - tmp)
            print "Difference: %f" % (difference)
            self.loss = tmp
            print "Loss: %f" % (self.loss)
            self.losses.append(self.loss)

    ####################
    # Helper Functions #
    ####################
    def pick_all_clusters(self):
        self.clusters = [[] for _ in range(self.k)]
        for image in self.data:
            self.pick_cluster(image)
        self.clusters = np.asarray(self.clusters)

    def pick_cluster(self, image):
        best_index = None
        best_distance = float("inf")
        for index in range(self.k):
            distance = np.linalg.norm(image - self.centroids[index])
            if distance < best_distance:
                best_distance = distance
                best_index = index
        self.clusters[best_index].append(image)

    def objective_function(self):
        total_loss = 0
        for i in range(self.k):
            mu = self.centroids[i]
            for image in self.clusters[i]:
                total_loss += np.linalg.norm(image - mu)
        return total_loss

    def compute_means(self):
        for i in range(self.k):
            self.centroids[i] = np.mean(self.clusters[i], axis=0)

    def visualize(self):
        for i in range(self.k):
            image = self.centroids[i].reshape((28, 28))
            imgplot = plt.imshow(image, cmap=cm.Greys)
            plt.show()

pdb.set_trace()

five = KMeans(images, 5)
print "Done with k=5"
ten = KMeans(images, 10)
print "Done with k=10"
twenty = KMeans(images, 20)
print "Done with k=20"
pdb.set_trace()
five.visualize()
pdb.set_trace()
ten.visualize()
pdb.set_trace()
twenty.visualize()
