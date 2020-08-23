import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math

import sklearn
# import sys
# sys.path.append('pca_coursera')

from ipywidgets import interact
from load_data import load_mnist
from distance_angle import distance

MNIST = load_mnist()
images = MNIST['data'].astype(np.double)
labels = MNIST['target'].astype(np.int)

distances = []
for i in range(len(images[:500])):
    for j in range(len(images[:500])):
        distances.append(distance(images[i], images[j]))

def show_img(first, second):
    plt.figure(figsize=(8,4))
    f = images[first].reshape(28, 28)
    s = images[second].reshape(28, 28)
    
    ax0 = plt.subplot2grid((2, 2), (0, 0))
    ax1 = plt.subplot2grid((2, 2), (1, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    
    #plt.imshow(np.hstack([f,s]), cmap='gray')
    ax0.imshow(f, cmap='gray')
    ax1.imshow(s, cmap='gray')
    ax2.hist(np.array(distances), bins=50)
    d = distance(f.ravel(), s.ravel())
    ax2.axvline(x=d, ymin=0, ymax=40000, color='C4', linewidth=4)
    ax2.text(0, 46000, "Distance is {:.2f}".format(d), size=12)
    ax2.set(xlabel='distance', ylabel='number of images')
    plt.show()

def most_similar_image():
    def show_img_d(first, second):
        # These comments are only necessary to pass Programming Assignment 2 in Coursera
        # from load_data import load_mnist
        # MNIST = load_mnist()
        # images = MNIST['data'].astype(np.double)

        f = images[first].reshape(28, 28)
        s = images[second].reshape(28, 28)
        d = distance(f.ravel(), s.ravel())
        return d
    """Find the index of the digit, among all MNIST digits
       that is the second-closest to the first image in the dataset (the first image is closest to itself trivially). 
       Your answer should be a single integer.
    """
    d = []
    for i in range(499):
        d.append(show_img_d(0,i))
    index = d.index(sorted(d)[1]) #<-- Change the -1 to the index of the most similar image.
    # You should do your computation outside this function and update this number
    # once you have computed the result
    return index

if __name__ == '__main__':
    result = most_similar_image()
    print(result)

