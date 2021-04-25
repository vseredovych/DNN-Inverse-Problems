import numpy as np

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def MSE(Y, T):
    return np.mean((Y - T) ** 2)
