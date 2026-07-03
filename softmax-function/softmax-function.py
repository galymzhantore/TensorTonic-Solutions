import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    stable = x - np.max(x, axis = -1, keepdims = True)
    expd = np.exp(stable)
    sums = np.sum(expd, axis = -1, keepdims = True)
    return expd / sums