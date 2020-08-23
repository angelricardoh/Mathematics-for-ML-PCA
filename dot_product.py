import numpy as np
import math

def length(x):
    """Compute the length of a vector"""
    length_x = math.sqrt(sum(np.square(x)))
    # N = len(x)
    # length_x = math.sqrt(sum(np.square(x)) / N)

    return length_x
  
x = np.array([3, 4])
y = np.array([-1, -1])
print(x)
print(y)
print(x.T @ y)
print(length(np.array([1,0])))