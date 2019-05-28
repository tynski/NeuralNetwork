import numpy as np
from layer import *


class FullyConnectedNet(object):
    """
    Fully connected nueral network with ReLu nonlinearities and softmax loss function.
    Network architecture look like this:
    input -> hiddenlayer with relu -> softmax -> output
    Learned parameters for Network are stored in self.params dictionary.
    """

    def __init__(self, hidden_dims, input_dim, num_classes, weight_scale=1e-2, reg=0.0, dtype=np.float32):
        """
        Initializer for new fully connected network.

        Inputs:
        hiddendims - a list of integers giving the size of each hidden layer.
        inputdim - an integer giving the size of the input.
        weight_scale - scalar giving std for weight initialization
        num_classes - an integer giving number of classe to classify
        """
        self.params = {}
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype

        # initialize the weights and biases
        for i in range(self.num_layers-1):
            self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale, (input_dim, hidden_dims[i]))
            self.params['b' + str(i + 1)] = np.zeros(hidden_dims[i])
            input_dim = hidden_dims[i]

        self.params['W' + str(self.num_layers)] = np.random.normal(0, weight_scale, (input_dim, num_classes))
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for fully connected network.

        Inputs:
        X - array of input data
        y - array of labels

        Returns:
        If y is None, then run test of model, returns:
        scores - classification scores

        If y is not None, then run train of model, returns:
        loss - scalar giving the loss
        gradient - dictionart mapping parameter names to gradients
        """
        X = X.astype(self.dtype)
        scores = None
        scores_cache = {}
        fc = None
        fc_cache = {}
        relu = None
        relu_cache = {}

        mode = 'test' if y is None else 'train'

        # compute scores through forwardpass
        for i in range(self.num_layers - 1):
            fc, fc_cache[str(i + 1)] = forward_pass(X, self.params['W' + str(i + 1)], self.params['b' + str(i + 1)])
            relu, relu_cache[str(i + 1)] = relu_forward(fc)
            X = relu

        scores, scores_cache = forward_pass(relu, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])

        # If test mode just return scores
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        loss, dsoftmax = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(np.square(self.params['W' + str(self.num_layers)])))

        # backward pass
        dx, dw, db = backward_pass(dsoftmax, scores_cache)

        # Store gradients of the last FC layer
        grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db

        for i in range(self.num_layers - 1, 0, -1):
            # backward pass for layer relu -> layer
            drelu = relu_backward(dx, relu_cache[str(i)])
            dx, dw, db = backward_pass(drelu, fc_cache[str(i)])

            grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db

            # Add regression for each layer
            loss += 0.5 * self.reg * (np.sum(np.square(self.params['W' + str(i)])))

        return loss, grads
