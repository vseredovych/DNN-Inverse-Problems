import numpy as np
import collections


class NeuralNetwork():
    def __init__(self):
        self.layers = None
        self.max_iterations = None
        self.batch_size = None
        self.learning_rate = None

    # Forward propagation step as a method.
    def forward_step(self, input_samples):
        """
        Compute and return the forward activation of each layer in layers.
        Input:
            input_samples: A matrix of input samples (each row 
                           is an input vector)
            layers: A list of Layers
        Output:
            A list of activations where the activation at each index 
            i+1 corresponds to the activation of layer i in layers. 
            activations[0] contains the input samples.  
        """
        # List of layer activations
        activations = [input_samples]
        # Compute the forward activations for each layer starting from the first
        X = input_samples

        for layer in self.layers:
            # Get the output of the current layer
            Y = layer.get_output(X)
            # Store the output for future processing
            activations.append(Y)
            # Set the current input as the activations of the previous layer
            X = activations[-1]
        return activations
    
    def backward_step(self, activations, targets):
        """
        Perform the backpropagation step over all the layers and return the parameter gradients.
        Input:
            activations: A list of forward step activations where the activation at 
                each index i+1 corresponds to the activation of layer i in layers. 
                activations[0] contains the input samples. 
            targets: The output targets of the output layer.
            layers: A list of Layers corresponding that generated the outputs in activations.
        Output:
            A list of parameter gradients where the gradients at each index corresponds to
            the parameters gradients of the layer at the same index in layers. 
        """
        # List of parameter gradients for each layer
        param_grads = collections.deque()

        # The error gradient at the output of the current layer
        output_grad = None

        # Propagate the error backwards through all the layers.
        for layer in reversed(self.layers):
            Y = activations.pop()

            # The output layer error is calculated different then hidden layer error.
            if output_grad is None:
                input_grad = layer.get_input_grad(Y, targets)
            else:
                # output_grad is not None (layer is not output layer)
                input_grad = layer.get_input_grad(Y, output_grad)

            # Get the input of this layer (activations of the previous layer)
            X = activations[-1]

            # Compute the layer parameter gradients used to update the parameters
            grads = layer.get_params_grad(X, output_grad)
            param_grads.appendleft(grads)

            # Compute gradient at output of previous layer (input of current layer):
            output_grad = input_grad
        return list(param_grads)

    def update_params(self, param_grads):
        """
        Function to update the parameters of the given layers with the given 
        gradients by gradient descent with the given learning rate.
        """
        for layer, layer_backprop_grads in zip(self.layers, param_grads):
            for param, grad in zip(layer.get_params_iter(), 
                                   layer_backprop_grads):
                # The parameter returned by the iterator point to the 
                # memory space of the original layer and can thus be 
                param -= self.learning_rate * grad

    def create_minibatches(self, X, Y):
        num_batches = X.shape[0] // self.batch_size

        # Create batches (X,Y) from the training set
        return list(zip(
            np.array_split(X, num_batches, axis=0),
            np.array_split(Y, num_batches, axis=0)))

    def fit(self, X, Y, X_test, Y_test, layers, 
            max_iterations=10, 
            batch_size=1024, 
            learning_rate=0.01,
            output_each_iter=None):
        
        self.layers = layers
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate        
        self.batch_costs = []
        self.train_costs = []
        self.val_costs = []
        
        for iteration in range(max_iterations):
            for X_mini, Y_mini in self.create_minibatches(X, Y):
                activations = self.forward_step(X_mini)

                batch_cost = layers[-1].get_cost(activations[-1], Y_mini)
                self.batch_costs.append(batch_cost)
                
                param_grads = self.backward_step(activations, Y_mini)

                self.update_params(param_grads)

            activations = self.forward_step(X)
            train_cost = layers[-1].get_cost(activations[-1], X)
            self.train_costs.append(train_cost)

            activations = self.forward_step(X_test)
            validation_cost = layers[-1].get_cost(activations[-1], X_test)

            self.val_costs.append(validation_cost)
            if output_each_iter and iteration % output_each_iter == 0:
                print(f"Iteration: {iteration}; Train loss: {train_cost}; Validation loss: {validation_cost};")

    def load_layers(self, layers):
        self.layers = layers
        