import scipy.io
import numpy as np
from scipy.sparse.linalg import svds
import csv

# Debugging
import time
import pdb

#################################################################################
# CLEAN UP / PREPROCESSING
M_JOKES = 100
N_USERS = 24983

DATA_DIR='/Users/David/dev/cs189/hw7/joke_data/'
data = scipy.io.loadmat(DATA_DIR+'joke_train.mat')

train_data = data['train']

validation = np.loadtxt(DATA_DIR+'validation.txt', dtype=int, delimiter=',')
validation_features = np.delete(validation, np.s_[2:], 1)
validation_labels = np.delete(validation, np.s_[:2], 1)

#################################################################################

class PCA():
    def __init__(self, data, d=20, zero=True, maxiter=1000, lam=1, eta=1e-9):
        # Dimensions
        self.d = d
        # Lambda for loss function
        self.lam = lam
        # Eta for learning rate
        self.eta = eta
        if zero:
            #Convert NaNs to Zeros
            self.data = np.nan_to_num(data)
            # PCA
            self.U, self.S, self.V = svds(self.data, k=d)
            # Get Weights
            self.u_reduced = self.transform(self.U.T, self.data).T # (d, 100) --> (100, d)
            self.v_reduced = self.transform(self.data, self.V.T) # (24983, d)
            # Build Ratings Matrix R
            self.R = np.dot(self.v_reduced, self.u_reduced.T) # (24983, 100)
        else:
            # Minimize using loss function
            self.data = data
            # Randomize ui, vj
            self.u_reduced = np.random.rand(self.data.shape[1], self.d)
            self.v_reduced = np.random.rand(self.data.shape[0], self.d)
            self.R = np.dot(self.v_reduced, self.u_reduced.T)
            # Use maxiter
            for iteration in range(maxiter):
                if iteration % 100 == 0:
                    print "Iteration %d / %d" % (iteration, maxiter)
                    # Get current loss
                    loss = self.loss_function(self.u_reduced, self.v_reduced, self.lam)
                    print "Loss %d" % (loss)
                # Calculate Gradients and Update
                dLdu = 2 * (np.dot(np.dot(self.u_reduced, self.v_reduced.T), self.v_reduced) - \
                        np.dot(np.nan_to_num(self.data.T), self.v_reduced) + \
                        self.lam * self.u_reduced)
                self.u_reduced -= (self.eta*dLdu)

                dLdv = 2 * (np.dot(np.dot(self.v_reduced, self.u_reduced.T), self.u_reduced) - \
                        np.dot(np.nan_to_num(self.data), self.u_reduced) + \
                        self.lam * self.v_reduced)
                self.v_reduced -= (self.eta*dLdv)
                #pdb.set_trace()
            self.R = np.dot(self.v_reduced, self.u_reduced.T)

    def transform(self, eigenvectors, data):
        return np.dot(eigenvectors, data)

    def predict(self, user, joke):
        return self.R[user-1][joke-1]

    def validate(self):
        predicted_labels = []
        for index in range(len(validation_features)):
            user, joke = validation_features[index]
            if self.predict(user, joke) > 0:
                label = 1
            else:
                label = 0
            predicted_labels.append([label])
        predicted_labels = np.array(predicted_labels)
        error = np.count_nonzero(validation_labels - predicted_labels) / float(len(validation_labels))
        accuracy = 1.0 - error
        print "Validation Accuracy: %f" % (accuracy)
        mse = self.mse()
        print "Mean Squared Error: %f" % (mse)
        total_loss = self.loss_function(self.u_reduced, self.v_reduced, self.lam)
        print "Loss: %f" % (total_loss)

    def mse(self):
        #mean_square_error = np.sum(np.square(self.R - self.data))
        mean_square_error = self.loss_function(self.u_reduced, self.v_reduced, 0)
        return mean_square_error

    def loss_function(self, ui, vj, l):
        term1 = np.nansum(np.square(np.dot(vj, ui.T) - self.data))
        term2 = l * np.nansum(np.square(ui))
        term3 = l * np.nansum(np.square(vj))
        return term1 + term2 + term3

#for eta in [1e-9, 1e-6, 1e-3]:
#    for lam in [0, 1e-3, 1e-2, 1e-1, 1, 2]:
#        print "==========PCA d=20 eta=%f lam=%d RATING==========" % (eta, lam)
#        #pca = PCA(train_data, d=d)
#        pca = PCA(train_data, d=20, zero=False, maxiter=2000, lam=lam, eta=eta)
#        pca.validate()

#for d in [2, 5, 10, 20]:
#    print "==========PCA d=%d RATING==========" % (d)
    #pca = PCA(train_data, d=20, zero=False, maxiter=10000, lam=0.1, eta=1e-9)
    #pca = PCA(train_data, d=d, zero=True, lam=0)
    #pca.validate()
pca = PCA(train_data, d=20, zero=False, maxiter=1000, lam=0.1, eta=1e-5)
pca.validate()

# Kaggle

#query = np.loadtxt(DATA_DIR+'query.txt', dtype=int, delimiter=',')
#query_features = np.delete(query, np.s_[2:], 1)
#labels = []
#for index in range(len(query_features)):
#    user, joke = query_features[index]
#    if pca.predict(user, joke) > 0:
#        label = 1
#    else:
#        label = 0
#    labels.append(label)
#labels = np.array(labels)
#
#with open('results.csv', 'wb') as csvfile:
#    writer = csv.writer(csvfile)
#    writer.writerow(['Id', 'Category'])
#    for i in range(len(labels)):
#        writer.writerow([i+1, labels[i]])

pdb.set_trace()
