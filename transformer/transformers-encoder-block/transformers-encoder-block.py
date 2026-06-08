import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis = -1, keepdims = True)
    var = np.var(x, axis = -1, keepdims = True)
    return gamma * (x - mean) / (np.sqrt(var + eps)) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    B, T_q, d_model = Q.shape
    T_k = K.shape[1]
    d_head = d_model // num_heads
    qw = (Q @ W_q).reshape(B, T_q, num_heads, d_head).transpose(0, 2, 1, 3)
    kw = (K @ W_k).reshape(B, T_k, num_heads, d_head).transpose(0, 2, 1, 3)
    vw = (V @ W_v).reshape(B, T_k, num_heads, d_head).transpose(0, 2, 1, 3)
    denom = qw @ kw.swapaxes(-2, -1)
    heads = softmax(denom / np.sqrt(d_head), axis = -1) @ vw # (B, H, T_q, d_head)
    return heads.transpose(0, 2, 1, 3).reshape(B, T_q, d_model) @ W_o
    
    
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    h = np.maximum(0, np.dot(x, W1) + b1)
    return np.dot(h, W2) + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    x_prime = layer_norm(x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads), gamma1, beta1)
    output = layer_norm(x_prime + feed_forward(x_prime, W1, b1, W2, b2), gamma2, beta2)
    return output