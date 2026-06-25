import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    xs = np.asarray(x)
    mean = np.mean(xs)
    n = xs.shape[0]
    var = np.sum((xs - mean) ** 2) / (n - 1)
    return var, np.sqrt(var)
