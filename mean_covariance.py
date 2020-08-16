import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('fivethirtyeight')
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
import time
import timeit
import math

from ipywidgets import interact

# GRADED FUNCTION: DO NOT EDIT THIS LINE
def mean_naive(X):
    "Compute the mean for a dataset X nby iterating over the data points"
    # X is of size (D,N) where D is the dimensionality and N the number of data points
    D, N = X.shape
    mean = np.zeros((D,1))
    for n in range(D):
        # Update the mean vector
        mean[n] = sum(X[n,:]) / N
        pass
    ###
    return mean

# Mine with for loop didn't work
def cov_naive(X):
    """Compute the covariance for a dataset of size (D,N) 
    where D is the dimension and N is the number of data points"""
    D, N = X.shape
    ### Edit the code below to compute the covariance matrix by iterating over the dataset.
    covariance = np.zeros((D, D))
    mean = mean_naive(X)
    ### Update covariance
    for n in range(D):
        covariance[n] = sum((X[n,:] - mean[n]) * ((X[n,:] - mean[n]).T)) / N
    ###
    
    # print(X[0,:] - mean[0])
    covariance[0][1] = sum((X[0,:] - mean[0]) * (X[1,:] - mean[1])) / N
    covariance[1][0] = sum((X[0,:] - mean[0]) * (X[1,:] - mean[1])) / N
    return np.cov(X, bias=True)


def cov_naive_2(X):
    """Compute the covariance for a dataset of size (D,N) 
    where D is the dimension and N is the number of data points"""
    D, N = X.shape
    covariance = np.zeros((D, D))

    temp = X - mean_naive(X)

    covariance = (temp @ temp.T)
    return covariance / N

def mean(X):
    "Compute the mean for a dataset of size (D,N) where D is the dimension and N is the number of data points"
    # given a dataset of size (D, N), the mean should be an array of size (D,1)
    # you can use np.mean, but pay close attention to the shape of the mean vector you are returning.
    D, N = X.shape
    ### Edit the code to compute a (D,1) array `mean` for the mean of dataset.
    mean = np.zeros((D,1))
    ### Update mean here
    for d in range(D):
        mean[d] = np.mean(X[d])
    ###
    return mean

def cov(X):
    "Compute the covariance for a dataset"
    # X is of size (D,N)
    # It is possible to vectorize our code for computing the covariance with matrix multiplications,
    # i.e., we do not need to explicitly
    # iterate over the entire dataset as looping in Python tends to be slow
    # We challenge you to give a vectorized implementation without using np.cov, but if you choose to use np.cov,
    # be sure to pass in bias=True.
    D, N = X.shape
    ### Edit the code to compute the covariance matrix
    covariance_matrix = np.zeros((D, D))
    ### Update covariance_matrix here
    
    ###
    return np.cov(X, bias=True)

if __name__ == '__main__':
    # First test
    # X_test = np.arange(6).reshape(2,3)
    # print(X_test)
    # print(mean(X_test))
    # print(mean_naive(X_test))
    # print(cov_naive(X_test))
    # print(cov(X_test))

    # Second test
    # Y_test = np.array([[1,5], [3,4]])
    # print(Y_test)
    # print(mean(Y_test))
    # print("""mean(Y_test)""")
    # print(mean_naive(Y_test))
    # print("""cov_naive(Y_test)""")
    # print(cov_naive(Y_test))
    # print("""cov_naive_2(Y_test)""")
    # print(cov_naive_2(Y_test))
    # print("""cov(Y_test)""")
    # print(cov(Y_test))

    image_shape = (64, 64)
    # Load faces data
    dataset = fetch_olivetti_faces('./')
    faces = dataset.data.T

    print('Shape of the faces dataset: {}'.format(faces.shape))
    print('{} data points'.format(faces.shape[1]))
    print(faces)
    np.testing.assert_almost_equal(mean(faces), mean_naive(faces), decimal=6)
    np.testing.assert_almost_equal(cov(faces), cov_naive_2(faces))