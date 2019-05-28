import numpy as np

"""
Utilities for computions between layers
"""

def forward_pass(x, w, b):
    """
    Inputs:
    x - numpy array containing input data
    w - numpy array containing weigts
    b - numpy array of biases

    Reurns:
    out - output
    cache - (x, w, b)
    """
    out = None
    out = np.dot(x, w) + b
    cache = (x, w, b)
    return out, cache


def backward_pass(dout, cache):
    """
    Inputs:
    dout - following derivative
    cache - stored variables from forward pass (x, w, b)

    Output:
    dx - gradient with respect of x
    dw - gradient with respect of w
    db - gradient with respect of b
    """
    x, w, _ = cache
    dx, dw, db = None, None, None

    dx = np.dot(dout, w.T)
    dx = np.reshape(dx, x.shape)
    dw = np.dot(x.T, dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Inputs:
    x - input

    Returns:
    out - output
    cahce - x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Inputs:
    dout - upstream derivatives
    cache - input x

    Returns:
    dx - gradient with respect to x
    """
    dx = None
    x = cache

    mask = (x >= 0)
    dx = dout * mask

    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    x - input data
    y - Vector of labels

    Returns a tuple of:
    loss - scalar giving the loss
    dx - radient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx




