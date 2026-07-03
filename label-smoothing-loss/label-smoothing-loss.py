import numpy as np
def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    preds = np.asarray(predictions)
    smooth_label = np.full(preds.shape, epsilon / len(predictions))
    smooth_label[target] += (1 - epsilon)
    return -np.sum(smooth_label * np.log(preds))