import numpy as np
from numba import jit


# Non-linear functions used
@jit
def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

@jit
def sigmoid_deriv(y):  # Derivative of logistic function
    return np.multiply(y, (1 - y))

@jit
def tanh(z):
    return np.tanh(z)
  
@jit
def tanh_deriv(y):
    return (1 - (np.tanh(y) ** 2))

def relu(z):
    return np.maximum(0.01*z, z)

def relu_deriv(y):
    y[y >= 0] = 1
    y[y < 0] = 0.01
    return y