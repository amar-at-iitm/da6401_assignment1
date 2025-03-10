import numpy as np
import wandb
import argparse
from tensorflow.keras.datasets import fashion_mnist, mnist 

# importing from local directory 
from propagation import forward_propagation, backpropagation
from optimizers import optimizers
from sweep_config import sweep_config

# Argument Parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", default="DA6401_assign_2")
    parser.add_argument("-we", "--wandb_entity", default="amar74384-iit-madras")
    parser.add_argument("-d", "--dataset", default="fashion-mnist", choices=["mnist", "fashion-mnist", "custom"])
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-l", "--loss", default="cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-o", "--optimizer", default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-w_i", "--weight_init", default="random", choices=["random", "Xavier"])
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
    parser.add_argument("-a", "--activation", default="sigmoid", choices=["identity", "sigmoid", "tanh", "ReLU"])
    return parser.parse_args()

# Dataset Loading and Preprocessing
def load_dataset(name):
    if name == "fashion-mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        return load_data("fashion-mnist.npz")  # Load from .npz file

    x_train, x_test = preprocess_data(x_train, y_train)
    y_train, y_test = np.eye(10)[y_train], np.eye(10)[y_test]  # One-hot encoding
    val_size = int(0.1 * x_train.shape[0])
    return (x_train[:-val_size], y_train[:-val_size]), (x_train[-val_size:], y_train[-val_size:]), (x_test, y_test)

def load_data(filepath):
    with np.load(filepath) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x, y):
    x = x.reshape(x.shape[0], -1) / 255.0  # Normalize and flatten
    y_one_hot = np.eye(10)[y]  # Convert labels to one-hot encoding
    return x, y_one_hot

# Xavier Initialization
def initialize_network(layer_sizes):
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
        weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1])))
        biases.append(np.zeros((1, layer_sizes[i + 1])))
    return weights, biases

# Compute Accuracy
def compute_accuracy(x, y, weights, biases, activation):
    activations, _ = forward_propagation(x, weights, biases, activation)
    predictions = np.argmax(activations[-1], axis=1)
    y_labels = np.argmax(y, axis=1)
    return np.mean(predictions == y_labels)

# Training Function
def train_network(x_train, y_train, x_val, y_val, layer_sizes, optimizer_name, epochs=10, batch_size=32, learning_rate=0.01, activation="sigmoid"):
    weights, biases = initialize_network(layer_sizes)
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Invalid optimizer '{optimizer_name}'. Available options: {list(optimizers.keys())}")
    
    optimizer = optimizers[optimizer_name](learning_rate)

    for epoch in range(epochs):
        indices = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[indices], y_train[indices]
        
        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            activations, z_values = forward_propagation(x_batch, weights, biases, activation)  # ✅ FIXED
            gradients_w, gradients_b = backpropagation(activations, z_values, weights, y_batch, activation)
            optimizer.update(weights, biases, gradients_w, gradients_b)
        
        train_activations, _ = forward_propagation(x_train, weights, biases, activation)  # ✅ FIXED
        loss = -np.mean(np.sum(y_train * np.log(train_activations[-1] + 1e-8), axis=1))
        val_acc = compute_accuracy(x_val, y_val, weights, biases, activation)
        
        wandb.log({"epoch": epoch + 1, "loss": loss, "val_acc": val_acc})
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Val Acc: {val_acc:.4f}")
    
    return weights, biases


# Main Function
def train():
    args = get_args()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    # Generating a meaningful run name using config values
    run_name = f"run_hl-{args.num_layers}_bs-{args.batch_size}_act-{args.activation}_opt-{args.optimizer}"
    wandb.run.name = run_name
    wandb.run.save()


    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset(args.dataset)
    layer_sizes = [x_train.shape[1]] + [args.hidden_size] * args.num_layers + [10]
    train_network(x_train, y_train, x_val, y_val, layer_sizes, args.optimizer, args.epochs, args.batch_size, args.learning_rate, args.activation)

if __name__ == "__main__":
    train()
