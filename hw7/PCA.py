import scipy.io
import numpy as np

import pdb

#################################################################################
# CLEAN UP / PREPROCESSING
M_JOKES = 100
N_USERS = 24983

DATA_DIR='/Users/David/dev/cs189/hw7/joke_data/'
data = scipy.io.loadmat(DATA_DIR+'joke_train.mat')

train_data = np.nan_to_num(data['train'])

validation = np.loadtxt(DATA_DIR+'validation.txt', dtype=int, delimiter=',')
validation_features = np.delete(validation, np.s_[2:], 1)
validation_labels = np.delete(validation, np.s_[:2], 1)

#################################################################################

class PCA():
    def __init__(self, data, d=20):
        self.data = data
        # Mean Vector
        self.mean = np.mean(data, axis=0)
        # Create Scatter Matrix
        print "Creating Scatter Matrix"
        self.scatter_matrix = np.zeros((self.mean.shape[0], self.mean.shape[0]))
        for i in range(self.data.shape[0]):
            val = (self.data[i] - self.mean).reshape(100,1)
            self.scatter_matrix += np.dot(val, val.T)
        print "Creating Covariance Matrix"
        tmp = []
        for i in range(100):
            tmp.append(self.data[:,i])
        self.cov_matrix = np.cov(tmp)
        # Get Eigenvectors
        eig_val_sc, eig_vec_sc = np.linalg.eig(self.scatter_matrix)
        #eig_val_cov, eig_vec_cov = np.linalg.eig(self.cov_matrix)
        self.eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
        # Get d dimensions
        self.eig_pairs.sort()
        self.eig_pairs.reverse()
        self.u_weight = np.array([self.eig_pairs[i][1] for i in range(d)])
        #self.v_weight = np.array([self.eig_pairs[:,i][1] for i in range(d)])

    def transform(self, data):
        return np.dot(data, self.weight.T)

test = PCA(train_data)

pdb.set_trace()
