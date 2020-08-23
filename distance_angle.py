import numpy as np
import math
import scipy
from scipy import spatial
import sklearn
from sklearn import metrics

def distance(x0, x1):
    """Compute distance between two vectors x0, x1 using the dot product"""
    # distance = x0.T @ x1
    diff = x0 - x1
    distance = math.sqrt(diff.T @ diff)
    return distance

def angle(x0, x1):
    """Compute the angle between two vectors x0, x1 using the dot product"""
    dot_product = x0.T @ x1
    x0_det = math.sqrt(x0.T @ x0)
    x1_det = math.sqrt(x1.T @ x1)
    angle = np.arccos(dot_product / (x0_det * x1_det))
    return angle


def pairwise_distance_matrix(X, Y):
    """Compute the pairwise distance between rows of X and rows of Y

    Arguments
    ----------
    X: ndarray of size (N, D)
    Y: ndarray of size (M, D)
    
    Returns
    --------
    distance_matrix: matrix of shape (N, M), each entry distance_matrix[i,j] is the distance between
    ith row of X and the jth row of Y (we use the dot product to compute the distance).
    """
    N, D = X.shape
    M, _ = Y.shape
    # distance_matrix = np.zeros((N, M)) # <-- EDIT THIS
    # for row in range(N):
    #     diff = X[row] - Y[row]
    #     distance_matrix[row] =  math.sqrt(diff.T @ diff)
    # return distance_matrix[:,0]

    # diff = X - Y
    # print(diff)
    # print(scipy.__version__)
    # distance_matrix = np.sqrt(sum(np.power(diff, 2)))

    # Y_extendend=np.vstack((Y for i in range(N)))
    # distance_matrix=scipy.spatial.distance.cdist(X,Y_extendend)
    
    # return distance_matrix[:,0]

    return sklearn.metrics.pairwise.euclidean_distances(X,Y)

if __name__ == '__main__':
    # X = np.array([1, 2])
    # Y = np.array([2, 1])

    # X = np.array([3, 4])
    # Y = np.array([-1, -1])

    # X = np.array([3, 4])
    # Y = np.array([1, -1])

    # X = np.array([1, 2, 3])
    # Y = np.array([-1, 0 , 8])

    # print('distance')
    # print(distance(X, Y))
    # print('angle')
    # print(angle(x,y))

    X = np.array([[1, 2], [2 ,3]])
    Y = np.array([[2, 1], [4, 1]])
    print(pairwise_distance_matrix(X, Y))