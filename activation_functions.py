import numpy as np


def identity_function(x):
    return np.array(x,dtype=np.float32)[0]


def sigmoid_function(x):
    x = np.array(x, dtype=np.float32)
    x = x.sum()
    return 1 / (1 + np.exp(-x))
