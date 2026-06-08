import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y = np.array(y_true)
    p = np.array(y_pred)
    correct_probs = p[np.arange(y.shape[0]), y]
    return -np.mean(np.log(correct_probs))