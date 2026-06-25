import numpy as np

def dropout(x, p=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=np.float32)
    keep_prob = 1.0 - p
    mask = (rng.random(x.shape) < keep_prob).astype(np.float32)
    mask /= keep_prob
    return x * mask, mask
    
    