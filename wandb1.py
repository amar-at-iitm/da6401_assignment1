import numpy as np
import wandb

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

# Initialize Network
def initialize_network(layer_sizes, init_method):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        if init_method == "xavier":
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
        else:  # Random initialization
            limit = 0.1
        weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1])))
        biases.append(np.zeros((1, layer_sizes[i + 1])))
    return weights, biases

# Activation Functions
def activation_fn(z, fn):
    if fn == "relu":
        return np.maximum(0, z)
    elif fn == "sigmoid":
        return 1 / (1 + np.exp(-z))
    elif fn == "tanh":
        return np.tanh(z)

def activation_derivative(z, fn):
    if fn == "relu":
        return (z > 0).astype(float)
    elif fn == "sigmoid":
        sig = 1 / (1 + np.exp(-z))
        return sig * (1 - sig)
    elif fn == "tanh":
        return 1 - np.tanh(z) ** 2

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability trick
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

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
def compute_accuracy(x, y, weights, biases, activation):
    activations, _ = forward_propagation(x, weights, biases, activation)
    predictions = np.argmax(activations[-1], axis=1)
    y_labels = np.argmax(y, axis=1)
    return np.mean(predictions == y_labels)

def train():
    wandb.init(entity="amar74384-iit-madras", project="DA6401_assign_1")
    config = wandb.config
    
    (x_train, y_train), (x_test, y_test) = load_data('fashion-mnist.npz')
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    
    layer_sizes = [784] + [config.hidden_size] * config.hidden_layers + [10]
    weights, biases = initialize_network(layer_sizes, config.weight_init)
    optimizer = SGD(config.learning_rate)
    

    wandb.define_metric("val_acc", summary="max")
    best_val_acc = 0  # Track best validation accuracy


    for epoch in range(config.epochs):
        indices = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[indices], y_train[indices]
        
        for i in range(0, x_train.shape[0], config.batch_size):
            x_batch = x_train[i:i + config.batch_size]
            y_batch = y_train[i:i + config.batch_size]
            activations, z_values = forward_propagation(x_batch, weights, biases, config.activation)
            gradients_w, gradients_b = backpropagation(activations, z_values, weights, y_batch, config.activation)
            optimizer.update(weights, biases, gradients_w, gradients_b)
        
        #loss = -np.mean(np.sum(y_train * np.log(activations[-1] + 1e-8), axis=1))
        loss = -np.mean(np.sum(y_batch * np.log(activations[-1] + 1e-8), axis=1))
        val_acc = compute_accuracy(x_test, y_test, weights, biases, config.activation)

        # Update best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        wandb.log({"epoch": epoch, "loss": loss, "val_acc": val_acc, "best_val_acc": best_val_acc})
        
       # wandb.log({"epoch": epoch, "loss": loss, "val_acc": val_acc})
    
    return weights, biases

sweep_config = {
    "method": "random",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [5, 10]},
        "hidden_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_init": {"values": ["random", "xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="DA6401_assign_1")
wandb.agent(sweep_id, function=train, count=10)
