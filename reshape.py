import numpy as np

def reshape(x):
  """return x_reshaped as a flattened vector of the multi-dimensional array x"""
  x_reshaped = np.reshape(x, 784)
  return x_reshaped

a = np.ones((28, 28))
print(reshape(a))