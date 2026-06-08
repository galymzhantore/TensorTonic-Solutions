import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def model_pred(X, w, b):
    return _sigmoid(np.dot(X, w) + b)
def BCE(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n, m = X.shape
    w = np.zeros(m)
    b = 0
    for epoch in range(steps):
        pred = model_pred(X, w, b)
        loss = BCE(pred, y)
        grad_w = np.mean(X.T @ (pred - y))
        grad_b = np.mean(pred - y)
        w -= lr*grad_w
        b -= lr*grad_b
    return w, b
        
        