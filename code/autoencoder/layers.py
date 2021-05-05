import numpy as np
import itertools
import collections
import autoencoder.activations as activations
import autoencoder.losses as losses

# Layers used in this model
class Layer(object):
    """Base class for the different layers.
    Defines base methods and documentation of methods."""
    
    def get_params_iter(self):
        """Return an iterator over the parameters (if any).
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place."""
        return []
    
    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters.
        The list has the same order as the get_params_iter iterator.
        X is the input.
        output_grad is the gradient at the output of this layer.
        """
        return []
    
    def get_output(self, X):
        """Perform the forward step linear transformation.
        X is the input."""
        pass
    
    def get_input_grad(self, Y, output_grad=None, T=None):
        """Return the gradient at the inputs of this layer.
        Y is the pre-computed output of this layer (not needed in 
        this case).
        output_grad is the gradient at the output of this layer 
         (gradient at input of next layer).
        Output layer uses targets T to compute the gradient based on the 
         output error instead of output_grad"""
        pass


class LinearLayer(Layer):
    """The linear layer performs a linear transformation to its input."""
    
    def __init__(self, n_in, n_out):
        """Initialize hidden layer parameters.
        n_in is the number of input variables.
        n_out is the number of output variables."""
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros(n_out)
        
    def get_params_iter(self):
        """Return an iterator over the parameters."""
        return itertools.chain(
            np.nditer(self.W, op_flags=['readwrite']),
            np.nditer(self.b, op_flags=['readwrite']))
    
    def get_output(self, X):
        """Perform the forward step linear transformation."""
        return (X @ self.W) + self.b
        
    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters."""
        # output_grad = dZ
        # X.T = Activation (A)
        JW = X.T @ output_grad
        Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(
            np.nditer(JW), np.nditer(Jb))]
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return output_grad @ self.W.T


class TanhLayer(Layer):
    def get_output(self, X):
        """Perform the forward step transformation."""
        return activations.tanh(X)
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return np.multiply(activations.tanh_deriv(Y), output_grad)


class SigmoidLayer(Layer):
    def get_output(self, X):
        """Perform the forward step transformation."""
        return activations.sigmoid(X)
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return np.multiply(activations.sigmoid_deriv(Y), output_grad)


class ReluLayer(Layer):
    def get_output(self, X):
        """Perform the forward step transformation."""
        return activations.relu(X)
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return np.multiply(activations.relu_deriv(Y), output_grad)

    
class SoftmaxOutputLayer(Layer):
    """The softmax output layer computes the classification 
    propabilities at the output."""
    
    def get_output(self, X):
        """Perform the forward step transformation."""
        return losses.softmax(X)
    
    def get_input_grad(self, Y, T):
        """Return the gradient at the inputs of this layer."""
        return (Y - T) / Y.shape[0]
    
    def get_cost(self, Y, T):
        """Return the cost at the output of this output layer."""
        return 1/2 * np.sum((Y - T)**2)


class MSEOutputLayer(Layer):
    """The softmax output layer computes the classification 
    propabilities at the output."""
    
    def get_output(self, X):
        """Perform the forward step transformation."""
        return X
    
    def get_input_grad(self, Y, T):
        """Return the gradient at the inputs of this layer."""
        return (Y - T) / Y.shape[0]
        
    def get_cost(self, Y, T):
        """Return the cost at the output of this output layer."""
        return losses.MSE(Y, T)
