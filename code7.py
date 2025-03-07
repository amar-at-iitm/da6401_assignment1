import numpy as np
import wandb
from optimizers import optimizers
from sklearn.metrics import confusion_matrix

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
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay  # L2 Regularization term
    
    def update(self, weights, biases, gradients_w, gradients_b):
        for i in range(len(weights)):
            weights[i] -= self.lr * (gradients_w[i] + self.weight_decay * weights[i])
            biases[i] -= self.lr * gradients_b[i]  # Biases are not regularized

def log_confusion_matrix(x_test, y_test, model_weights, model_biases, activation):
    # Get model predictions
    activations, _ = forward_propagation(x_test, model_weights, model_biases, activation)
    y_pred = np.argmax(activations[-1], axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Log confusion matrix to WandB
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                               y_true=y_true, 
                                                               preds=y_pred,
                                                               class_names=[str(i) for i in range(10)])})

# Training Function
def compute_accuracy(x, y, weights, biases, activation):
    activations, _ = forward_propagation(x, weights, biases, activation)
    predictions = np.argmax(activations[-1], axis=1)
    y_labels = np.argmax(y, axis=1)
    return np.mean(predictions == y_labels)

def train():
    wandb.init(entity="amar74384-iit-madras", project="DA6401_assign_2")

    config = wandb.config

    # Generate a meaningful run name using config values
    run_name = f"run_hl-{config.hidden_layers}_bs-{config.batch_size}_act-{config.activation}_opt-{config.optimizer}"
    
    # Set the name dynamically after initializing wandb
    wandb.run.name = run_name
    wandb.run.save()
    
    (x_train, y_train), (x_test, y_test) = load_data('fashion-mnist.npz')
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    
    # Split 10% of training data as validation set
    val_split = int(0.1 * x_train.shape[0])
    x_val, y_val = x_train[:val_split], y_train[:val_split]
    x_train, y_train = x_train[val_split:], y_train[val_split:]

    # Initialize network
    layer_sizes = [784] + [config.hidden_size] * config.hidden_layers + [10]
    weights, biases = initialize_network(layer_sizes, config.weight_init)

    # Select optimizer
    #optimizer_class = optimizers.get(config.optimizer, optimizers["sgd"])

    optimizer_class = optimizers.get(config.optimizer, SGD)  # Default to SGD if not found
    optimizer = optimizer_class(config.learning_rate, weight_decay=config.weight_decay)

    for epoch in range(config.epochs):
        indices = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[indices], y_train[indices]
        
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for i in range(0, x_train.shape[0], config.batch_size):
            x_batch = x_train[i:i + config.batch_size]
            y_batch = y_train[i:i + config.batch_size]
            
            activations, z_values = forward_propagation(x_batch, weights, biases, config.activation)
            gradients_w, gradients_b = backpropagation(activations, z_values, weights, y_batch, config.activation)
            optimizer.update(weights, biases, gradients_w, gradients_b)
            
            batch_loss = -np.mean(np.sum(y_batch * np.log(activations[-1] + 1e-8), axis=1))
            epoch_loss += batch_loss * x_batch.shape[0]
            correct_predictions += np.sum(np.argmax(activations[-1], axis=1) == np.argmax(y_batch, axis=1))
            total_samples += x_batch.shape[0]
        

        # Compute training loss
        train_activations, _ = forward_propagation(x_train, weights, biases, config.activation)
        train_loss = -np.mean(np.sum(y_train * np.log(train_activations[-1] + 1e-8), axis=1))
        train_acc = compute_accuracy(x_train, y_train, weights, biases, config.activation)

        # Compute validation loss
        val_activations, _ = forward_propagation(x_val, weights, biases, config.activation)
        val_loss = -np.mean(np.sum(y_val * np.log(val_activations[-1] + 1e-8), axis=1))
        val_acc = compute_accuracy(x_val, y_val, weights, biases, config.activation)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        # Log confusion matrix after training
        log_confusion_matrix(x_test, y_test, weights, biases, config.activation)
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

sweep_id = wandb.sweep(sweep_config, project="DA6401_assign_2")
wandb.agent(sweep_id, function=train, count=2)