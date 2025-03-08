import numpy as np

def activation_fn(z, fn):
    if fn == "relu":
        return np.maximum(0, z)
    elif fn == "sigmoid":
        return 1 / (1 + np.exp(-z))
    elif fn == "tanh":
        return np.tanh(z)
    else:
        raise ValueError("Unsupported activation function")

def activation_derivative(z, fn):
    if fn == "relu":
        return (z > 0).astype(float)
    elif fn == "sigmoid":
        sig = 1 / (1 + np.exp(-z))
        return sig * (1 - sig)
    elif fn == "tanh":
        return 1 - np.tanh(z) ** 2
    else:
        raise ValueError("Unsupported activation function")

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability trick
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
