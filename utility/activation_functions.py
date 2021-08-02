import numpy as np


def identity_function(x):
    return np.array(x,dtype=np.float32)[0]


def sigmoid_function(x):
    x = np.array(x, dtype=np.float32)
    x = x.sum()
    return 1 / (1 + np.exp(-x))


def unsigned_step_function(x):
    x = np.array(x, dtype=np.float32)
    x = x.sum()
    x = 1.0 * (x > 0.0)
    return x


def sin_function(x):
    x = np.array(x, dtype=np.float32)
    x = x.sum()
    return np.sin(np.pi*x)


def gauss_function(x):
    x = np.array(x, dtype=np.float32)
    x = x.sum()
    return np.exp(-np.multiply(x, x) / 2.0)


def tanh_function(x):
    x = np.array(x, dtype=np.float32)
    x = x.sum()
    return np.tanh(x)


def inverse_function(x):
    x = np.array(x, dtype=np.float32)
    x = x.sum()
    return -x


def abs_function(x):
    x = np.array(x, dtype=np.float32)
    x = x.sum()
    return np.abs(x)


def relu_function(x):
    x = np.array(x, dtype=np.float32)
    x = x.sum()
    return np.maximum(x,0)


def cosine_function(x):
    x = np.array(x, dtype=np.float32)
    x = x.sum()
    return np.cos(np.pi*x)


def squared_function(x):
    x = np.array(x, dtype=np.float32)
    x = x.sum()
    return np.power(x,2)


all_activation_functions = [identity_function, sigmoid_function, unsigned_step_function,
                            sin_function, gauss_function, tanh_function, inverse_function,
                            abs_function, relu_function, cosine_function, squared_function]