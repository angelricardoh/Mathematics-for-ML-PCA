import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('fivethirtyeight')
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
import time
import timeit

from mean_covariance import mean
from mean_covariance import cov

# GRADED FUNCTION: DO NOT EDIT THIS LINE
def affine_mean(mean, A, b):
    """Compute the mean after affine transformation
    Args:
        x: ndarray, the mean vector
        A, b: affine transformation applied to x
    Returns:
        mean vector after affine transformation
    """
    ### Edit the code below to compute the mean vector after affine transformation
    affine_m = np.zeros(mean.shape) # affine_m has shape (D, 1)
    ### Update affine_m
    affine_m = (A @ mean) + b
    ###
    return affine_m

def affine_covariance(S, A, b):
    """Compute the covariance matrix after affine transformation
    Args:
        S: ndarray, the covariance matrix
        A, b: affine transformation applied to each element in X        
    Returns:
        covariance matrix after the transformation
    """
    ### EDIT the code below to compute the covariance matrix after affine transformation
    affine_cov = np.zeros(S.shape) # affine_cov has shape (D, D)
    ### Update affine_cov
    affine_cov = A @ S @ A.T
    ###
    return affine_cov

if __name__ == '__main__':
    random = np.random.RandomState(42)
    A = random.randn(4,4)
    b = random.randn(4,1)

    X = random.randn(4,100) # D = 4, N = 100

    X1 = (A @ X) + b  # applying affine transformation to each sample in X
    X2 = (A @ X1) + b # twice

    # # print(X1)
    # print(mean(X1))
    # print(affine_mean(mean(X), A, b))

    np.testing.assert_almost_equal(mean(X1), affine_mean(mean(X), A, b))
    np.testing.assert_almost_equal(cov(X1),  affine_covariance(cov(X), A, b))

    # print(cov(X2))
    # print(affine_covariance(cov(X1), A, b))

    np.testing.assert_almost_equal(mean(X2), affine_mean(mean(X1), A, b))
    np.testing.assert_almost_equal(cov(X2),  affine_covariance(cov(X1), A, b))