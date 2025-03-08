import numpy as np 
from activation import activation_fn, activation_derivative, softmax
# Forward Propagation
def forward_propagation(x, weights, biases, activation):
    activations = [x]
    z_values = []
    
    for i in range(len(weights) - 1):
        z = activations[-1] @ weights[i] + biases[i]
        z_values.append(z)
        activations.append(activation_fn(z, activation))
    
    z = activations[-1] @ weights[-1] + biases[-1]
    z_values.append(z)
    activations.append(softmax(z))
    
    return activations, z_values

# Backpropagation
def backpropagation(activations, z_values, weights, y_true, activation):
    gradients_w = [None] * len(weights)
    gradients_b = [None] * len(weights)
    
    error = activations[-1] - y_true
    
    for i in reversed(range(len(weights))):
        gradients_w[i] = activations[i].T @ error / y_true.shape[0]
        gradients_b[i] = np.mean(error, axis=0, keepdims=True)
        if i > 0:
            error = (error @ weights[i].T) * activation_derivative(z_values[i - 1], activation)
    
    return gradients_w, gradients_b

