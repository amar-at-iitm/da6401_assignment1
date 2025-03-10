import numpy as np
import wandb
from sklearn.metrics import confusion_matrix

# importing from local directory
from optimizers import optimizers
from sweep_config import sweep_config
from propagation import forward_propagation, backpropagation


# Loading and Preprocessing Fashion-MNIST Dataset
def load_data(filepath):
    with np.load(filepath) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x, y):
    x = x.reshape(x.shape[0], -1) / 255.0  # Normalizing and flattening
    y_one_hot = np.eye(10)[y]  # Converting labels to one-hot encoding
    return x, y_one_hot

# Initializing The Network
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

    # Generating a meaningful run name using config values
    run_name = f"run_hl-{config.hidden_layers}_bs-{config.batch_size}_act-{config.activation}_opt-{config.optimizer}"
    wandb.run.name = run_name
    wandb.run.save()
    
    (x_train, y_train), (x_test, y_test) = load_data('fashion-mnist.npz')
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    
    # Splitting 10% of training data as validation set
    val_split = int(0.1 * x_train.shape[0])
    x_val, y_val = x_train[:val_split], y_train[:val_split]
    x_train, y_train = x_train[val_split:], y_train[val_split:]

    # Initializing the network
    layer_sizes = [784] + [config.hidden_size] * config.hidden_layers + [10]
    weights, biases = initialize_network(layer_sizes, config.weight_init)

    # Selecting optimizer
    optimizer_class = optimizers.get(config.optimizer, optimizers["sgd"])
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
        
        # Computing training loss
        train_activations, _ = forward_propagation(x_train, weights, biases, config.activation)
        #####################################################################################
        # Compute training loss with L2 regularization (Squared Error Loss)
        train_loss = np.mean(np.sum((y_train - train_activations[-1])**2, axis=1)) + \
             (config.weight_decay / 2) * sum(np.sum(w**2) for w in weights)
        #######################################################################################
        train_acc = compute_accuracy(x_train, y_train, weights, biases, config.activation)

        # Computing validation loss
        val_activations, _ = forward_propagation(x_val, weights, biases, config.activation)
        #################################################################################
        # Compute validation loss with L2 regularization (Squared Error Loss)
        val_loss = np.mean(np.sum((y_val - val_activations[-1])**2, axis=1)) + \
           (config.weight_decay / 2) * sum(np.sum(w**2) for w in weights)
        #################################################################################  
        val_acc = compute_accuracy(x_val, y_val, weights, biases, config.activation)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})


    return weights, biases

sweep_id = wandb.sweep(sweep_config, project="DA6401_assign_2")
wandb.agent(sweep_id, function=train, count=2)