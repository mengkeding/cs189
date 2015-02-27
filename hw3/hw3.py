import numpy as np
import matplotlib.pyplot as plt
import math

# Problem 1
mu1, sigma1 = 3, 9**0.5
x1 = sigma1 * np.random.randn(100) + mu1

mu2, sigma2 = 4, 4**0.5
x2 = sigma2 * np.random.randn(100) + mu2 + (0.5 * x1)

# Calculate covariance matrix
arr = np.array([x1, x2])
cov = np.cov(arr)

# Eigenvectors
# w = eigenvalues
# v = eigenvectors
w, v = np.linalg.eig(cov)

eig1 = w[0] * v[0]
eig2 = w[1] * v[1]

a = np.array([eig1, eig2]).T

sort_eig = sorted([(w[0], v[0]), (w[1], v[1])], key=lambda tup: tup[0])
u = np.concatenate(([sort_eig[1][1]], [sort_eig[0][1]]))

plt.scatter(x1, x2)
plt.quiver([3,3], [4,4], a[0], a[1])
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.show()

centered = np.array([x1 - mu1, x2 - mu2])
rotated = np.dot(u.T, centered)
plt.scatter(rotated[0], rotated[1])
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.show()
#import pdb; pdb.set_trace()


