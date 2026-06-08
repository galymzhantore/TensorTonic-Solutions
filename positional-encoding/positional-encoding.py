import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]
    idx = np.arange(0, d_model, 2)
    values = pos / (base ** (idx / d_model))
    pe[:, 0::2] = np.sin(values)
    pe[:, 1::2] = np.cos(values[:, :(d_model // 2)])
    return pe
    