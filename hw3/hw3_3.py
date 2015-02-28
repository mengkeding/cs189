import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

def isocontour(mu1, sigma1, mu2=None, sigma2=None):
    x = np.arange(-10.0, 10.0, 0.1)
    y = np.arange(-10.0, 10.0, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X, Y, sigma1[0,0], sigma1[1,1], mu1[0], mu1[1], sigma1[0,1])
    if not (mu2 == None and sigma2 == None):
        Z2 = mlab.bivariate_normal(X, Y, sigma2[0,0], sigma2[1,1], mu2[0], mu2[1], sigma2[0,1])
        Z = Z - Z2
    plt.contour(X,Y,Z)
    plt.show()

# Part A
m = np.array([1, 1])
s = np.array([[2, 0],
            [0, 1]])
isocontour(m, s)

# Part B
m = np.array([-1, 2])
s = np.array([[3, 1],
            [1, 2]])
isocontour(m, s)


# Part C
m1 = np.array([0, 2])
m2 = np.array([2, 0])
s1 = np.array([[1, 1],
            [1, 2]])
s2 = np.array([[1, 1],
            [1, 2]])
isocontour(m1, s1, m2, s2)

# Part D
m1 = np.array([0, 2])
m2 = np.array([2, 0])
s1 = np.array([[1, 1],
            [1, 2]])
s2 = np.array([[3, 1],
            [1, 2]])
isocontour(m1, s1, m2, s2)

# Part E
m1 = np.array([1, 1])
m2 = np.array([-1, 1])
s1 = np.array([[1, 0],
            [0, 2]])
s2 = np.array([[2, 1],
            [1, 2]])
isocontour(m1, s1, m2, s2)
#import pdb; pdb.set_trace()
