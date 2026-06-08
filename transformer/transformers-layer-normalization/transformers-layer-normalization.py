import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    mean = np.mean(x, axis = -1, keepdims = True)
    std = np.std(x, axis = -1, keepdims = True)
    normalized = gamma * (x - mean) / np.sqrt(std ** 2 + eps)
    normalized += beta
    return normalized