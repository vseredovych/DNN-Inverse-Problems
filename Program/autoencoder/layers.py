class TanhLayer(Layer, activation):
    def get_output(self, X):
        """Perform the forward step transformation."""
        return tanh(X)
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return np.multiply(logistic_deriv(Y), output_grad)

class SigmoidLayer(Layer, activation):
    def get_output(self, X):
        """Perform the forward step transformation."""
        return sigmoid(X)
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return np.multiply(logistic_deriv(Y), output_grad)

class ReluLayer(Layer, activation):
    def get_output(self, X):
        """Perform the forward step transformation."""
        pass
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return np.multiply(tanh_deriv(Y), output_grad)