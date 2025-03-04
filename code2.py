import numpy as np
import pandas as pd

# Load Fashion-MNIST dataset from local directory
def load_data(filepath):
    with np.load(filepath) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
    return (x_train, y_train), (x_test, y_test)

# Normalize and flatten the images
def preprocess_data(x):
    return x.reshape(x.shape[0], -1) / 255.0  # Flatten and normalize

# Initialize weights and biases
def initialize_network(layer_sizes):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
        biases.append(np.zeros((1, layer_sizes[i + 1])))
    return weights, biases

# Activation functions
def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability trick
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Forward propagation
def forward_propagation(x, weights, biases):
    for i in range(len(weights) - 1):
        x = relu(np.dot(x, weights[i]) + biases[i])
    return softmax(np.dot(x, weights[-1]) + biases[-1])

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = load_data('fashion-mnist.npz')
x_train, x_test = preprocess_data(x_train), preprocess_data(x_test)

# Define network architecture with default values
use_default = input("Use default layer sizes [784, 128, 64, 10]? (y/n): ").strip().lower()
if use_default == 'n':
    layer_sizes = list(map(int, input("Enter layer sizes separated by spaces: ").split()))
else:
    layer_sizes = [784, 128, 64, 10]  # Input layer, hidden layers, output layer

# Initialize network
weights, biases = initialize_network(layer_sizes)

# Get probability distribution for first 5 test images
sample_images = x_test[:5]
predictions = forward_propagation(sample_images, weights, biases)

# Convert to pandas DataFrame for better readability
probability_df = pd.DataFrame(predictions, columns=[f'Class {i}' for i in range(10)])
print(probability_df)
