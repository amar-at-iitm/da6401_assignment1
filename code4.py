import numpy as np

# Load and Preprocess Fashion-MNIST Dataset
def load_data(filepath):
    with np.load(filepath) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x, y):
    x = x.reshape(x.shape[0], -1) / 255.0  # Normalize and flatten
    y_one_hot = np.eye(10)[y]  # Convert labels to one-hot encoding
    return x, y_one_hot

# Initialize Network with Xavier Initialization
def initialize_network(layer_sizes):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
        weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1])))
        biases.append(np.zeros((1, layer_sizes[i + 1])))
    return weights, biases

# Activation Functions
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability trick
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Forward Propagation
def forward_propagation(x, weights, biases):
    activations = [x]
    z_values = []
    
    for i in range(len(weights) - 1):
        z = activations[-1] @ weights[i] + biases[i]
        z_values.append(z)
        activations.append(relu(z))
    
    z = activations[-1] @ weights[-1] + biases[-1]
    z_values.append(z)
    activations.append(softmax(z))
    
    return activations, z_values

# Backpropagation
def backpropagation(activations, z_values, weights, y_true):
    gradients_w = [None] * len(weights)
    gradients_b = [None] * len(weights)
    
    error = activations[-1] - y_true
    
    for i in reversed(range(len(weights))):
        gradients_w[i] = activations[i].T @ error / y_true.shape[0]
        gradients_b[i] = np.mean(error, axis=0, keepdims=True)
        if i > 0:
            error = (error @ weights[i].T) * relu_derivative(z_values[i - 1])
    
    return gradients_w, gradients_b

# Optimizers
class Optimizer:
    def update(self, weights, biases, gradients_w, gradients_b):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def update(self, weights, biases, gradients_w, gradients_b):
        for i in range(len(weights)):
            weights[i] -= self.lr * gradients_w[i]
            biases[i] -= self.lr * gradients_b[i]

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v_w = None
        self.v_b = None
    
    def update(self, weights, biases, gradients_w, gradients_b):
        if self.v_w is None:
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]
        
        for i in range(len(weights)):
            self.v_w[i] = self.momentum * self.v_w[i] - self.lr * gradients_w[i]
            self.v_b[i] = self.momentum * self.v_b[i] - self.lr * gradients_b[i]
            weights[i] += self.v_w[i]
            biases[i] += self.v_b[i]

optimizers = {
    "sgd": SGD,
    "momentum": Momentum,
}

# Training Function
def compute_accuracy(x, y, weights, biases):
    activations, _ = forward_propagation(x, weights, biases)
    predictions = np.argmax(activations[-1], axis=1)
    y_labels = np.argmax(y, axis=1)
    return np.mean(predictions == y_labels)

def train_network(x_train, y_train, x_val, y_val, layer_sizes, optimizer_name, epochs=10, batch_size=32, learning_rate=0.01):
    weights, biases = initialize_network(layer_sizes)
    optimizer = optimizers[optimizer_name](learning_rate)
    
    for epoch in range(epochs):
        indices = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[indices], y_train[indices]
        
        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            activations, z_values = forward_propagation(x_batch, weights, biases)
            gradients_w, gradients_b = backpropagation(activations, z_values, weights, y_batch)
            optimizer.update(weights, biases, gradients_w, gradients_b)
        
        #loss = -np.mean(np.sum(y_train * np.log(activations[-1] + 1e-8), axis=1))
        train_activations, _ = forward_propagation(x_train, weights, biases)
        loss = -np.mean(np.sum(y_train * np.log(train_activations[-1] + 1e-8), axis=1))

        val_acc = compute_accuracy(x_val, y_val, weights, biases)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Val Acc: {val_acc:.4f}")
    
    return weights, biases

# Load and Preprocess Data
(x_train, y_train), (x_test, y_test) = load_data('fashion-mnist.npz')
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

# Ask User for Optimizer
while True:
    optimizer_name = input(f"Select an optimizer {list(optimizers.keys())}: ").strip().lower()
    if optimizer_name in optimizers:
        break
    print("Invalid choice. Please select from the available options.")

# Train the Network
train_network(x_train, y_train, x_test, y_test, [784, 128, 64, 10], optimizer_name)
