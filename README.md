# DA6401_assignment1
#### `Amar Kumar`  `(MA24M002)`
#### `M.Tech (Industrial Mathematics and Scientific Computing)` `IIT Madras`


## Problem Statement 
In this assignment, you need to implement a feedforward neural network and write the backpropagation code for training the network. We strongly recommend using numpy for all matrix/vector operations. You are not allowed to use any automatic differentiation packages. This network will be trained and tested using the Fashion-MNIST dataset. Specifically, given an input image (28 x 28 = 784 pixels) from the Fashion-MNIST dataset, the network will be trained to classify the image into 1 of 10 classes.

**Your code will have to follow the format specified in the Code Specifications section.**



# Fashion-MNIST Neural Network from Scratch

This assignment implements a **fully connected neural network** for **Fashion-MNIST classification** using only **NumPy, Pandas**. The code is modular, optimized using **WandB sweeps**, and follows best software engineering practices.
### Key Features
- **Custom Neural Network Implementation:** Built from scratch with forward propagation, backpropagation, and optimizers.
- **Hyperparameter Optimization:** Utilized **WandB sweeps** to identify the best model.
- **Loss Function Comparison:** Evaluated **cross-entropy loss vs. squared error loss**.
- **Confusion Matrix Visualization:** Automatically logs confusion matrics for **Best Run** to **WandB**.
- **Proper Data Handling:** Ensured **randomized train-test splits** and ethical ML practices.

### File Structure
- `code1.ipynb` – Tests the best model and plots the confusion matrix.
- `code2.ipynb` – Compares cross-entropy loss with squared error loss.
- `code3.py` – Tests the best model and plots the confusion matrix.
- `code4.py` – Compares cross-entropy loss with squared error loss.  
- `code7.py` – Tests the best model and plots the confusion matrix.
- `code8.py` – Compares cross-entropy loss with squared error loss.
- `train.py` – Main script for training the best model.
- `optimizers.py` – Implementations of various optimizers.
- `propagation.py` – Forward and backward propagation functions.
- `sweep_config.py` – Configuration file for hyperparameter tuning.
- `best_run.py` – Stores the best hyperparameter configuration.
- `fashion-mnist.npz` – The dataset used for training and evaluation.


### Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/amar-at-iitm/da6401_assignment1
   cd da6401_assignment1
   ```
2. Install required dependencies:
   ```bash
   pip install numpy wandb tensorflow
   ```
3. Configure WandB:
   ```bash
   wandb login
   ```
### Question 1: Fashion-MNIST Dataset Visualization 
`code1.ipynb` loads the Fashion-MNIST dataset using Keras, saves it as `.npz`, and visualizes one sample image per class in a 2x5 grid. It also logs the visualization to Weights & Biases (WandB).

#### Dataset Overview
Fashion-MNIST consists of 70,000 grayscale images (28x28 pixels) and **The dataset is divided into two parts 60,000 for training and 10,000 for testing**. The dataset consists of 10 clothing categories:
- T-shirt/top, Trouser, Pullover, Dress, Coat
- Sandal, Shirt, Sneaker, Bag, Ankle boot



### Question 2: Feedforward Neural Network for Fashion-MNIST 
`code2.ipynb` implements a flexible feedforward neural network that takes Fashion-MNIST images as input and outputs a probability distribution over 10 classes.

- Supports user-defined or default layer sizes (`[784, 128, 64, 10]`).
- Uses ReLU activation for hidden layers and Softmax for the output layer.
- Accepts an image index (1-70,000) from the user and predicts class probabilities.

### Steps in the notebook
1. **Load and Preprocess Data**: Loads `fashion-mnist.npz`, flattens images, and normalizes pixel values.
2. **Initialize Neural Network**: Creates weight matrices and bias vectors for a customizable layer structure.
3. **Forward Propagation**: Applies ReLU activation for hidden layers and Softmax for the output.
4. **User Interaction**: Allows selection of network architecture and an image index.
5. **Prediction**: Computes class probabilities and displays them using pandas DataFrame.


### Question 3: Fashion-MNIST Feedforward Neural Network with Backpropagation

`code3.py` implements a customizable feedforward neural network (FFNN) with backpropagation to classify Fashion-MNIST images. The model supports multiple optimization algorithms and allows flexibility in layer configuration and batch sizes.

- **Customizable architecture:** Easily modify the number of layers and neurons.
- **Backpropagation:** Implemented with gradient computation.
- **Optimizers:** Supports multiple optimization methods:
  - Stochastic Gradient Descent (SGD)
  - Momentum-based Gradient Descent
  - Nesterov Accelerated Gradient Descent
  - RMSprop
  - Adam
  - Nadam
- **Batch Training:** Supports mini-batch updates.

#### Command to Run
```terminal
python code3.py
```

#### Implementation Details
##### Data Handling
- Loads Fashion-MNIST dataset from a `.npz` file.
- Normalizes pixel values to [0,1].
- Converts labels to one-hot encoding.

##### Neural Network
- Uses ReLU activation for hidden layers and softmax for output.
- Forward propagation computes activations layer-by-layer.
- Backpropagation updates weights using gradients.
- Supports Xavier initialization for weight stability.

##### Training Process
- Shuffles training data at each epoch.
- Uses batch-wise forward and backward propagation.
- Loss function: **Cross-entropy loss**.
- Monitors validation accuracy at each epoch.

##### Customization
Modify layer sizes interactively or by editing the following line in `code3.py`:
```python
train_network(x_train, y_train, x_test, y_test, [784, 128, 64, 10], optimizer_name)
```




### Question 4,5,6: Hyperparameter Tuning with WandB Sweeps

`code4.py` implements hyperparameter tuning using Weights & Biases (WandB) sweeps to optimize a neural network for Fashion-MNIST classification. The network is trained using different optimization algorithms, activation functions, and weight initializations, allowing for an efficient search of the best performing model.

- Supports multiple optimizers: SGD, Momentum, Nesterov, RMSprop, Adam, Nadam
- Configurable hyperparameters: number of hidden layers, layer sizes, learning rate, weight decay, batch size, and activation function
- Uses WandB to log and visualize training performance
- Performs a hyperparameter sweep with a defined search strategy
- Splits 10% of training data as a validation set

#### Hyperparameter Tuning Strategy
The hyperparameters are tuned using the `wandb.sweep()` functionality. Given the large search space, we use **Bayesian Optimization** as our search strategy to efficiently explore promising configurations while minimizing redundant trials.

#### Hyperparameters Considered:
- **Epochs:** 5, 10
- **Hidden Layers:** 3, 4, 5
- **Hidden Layer Size:** 32, 64, 128
- **Weight Decay (L2 Regularization):** 0, 0.0005, 0.5
- **Learning Rate:** 1e-3, 1e-4
- **Optimizer:** SGD, Momentum, Nesterov, RMSprop, Adam, Nadam
- **Batch Size:** 16, 32, 64
- **Weight Initialization:** Random, Xavier
- **Activation Functions:** Sigmoid, Tanh, ReLU

#### Network Initialization
- Fully connected neural network initialized with user-defined parameters.
- Supports both Xavier and random initialization.

#### Training Pipeline
- Forward propagation computes activations for each layer.
- Backpropagation computes gradients and updates weights using the chosen optimizer.
- Training loss includes L2 regularization (if weight decay is applied).
- Training and validation performance is logged using WandB.

#### WandB Integration
- Each run is uniquely named based on key hyperparameters (e.g., `run_hl-3_bs-16_act-tanh_opt-adam`).
- Sweeps are created using `wandb.sweep()` with `count=500` to explore 500 configurations.
- Training performance metrics (loss, accuracy) are logged and visualized.

#### Running the Sweep
```terminal
python code3.py
```
It will automatically handle everything.

#### Observations & Insights
- The parallel coordinates plot helps identify parameter combinations that perform well.
- The correlation summary shows the relationship between hyperparameters and model accuracy.
- Models with ReLU activation and Xavier initialization generally perform better.
- Larger batch sizes (32, 64) with Adam or RMSprop optimize convergence speed and stability.
- A learning rate of `1e-3` works well with Adam, while `1e-4` is preferred for SGD-based optimizers.

#### Recommended Configuration
I am not able to achieve **95% accuracy**, but got **88.8% accuracy** with the following setup:

- **activation:"sigmoid"**
- **batch_size:32**
- **epochs:10**
- **hidden_layers:3**
- **hidden_size:128**
- **learning_rate:0.001**
- **optimizer:"adam"**
- **weight_decay:0**
- **weight_init:"random"**



### Question 7: Confusion Matrix for Best Run
`code7.py` logs, the confusion matrix for the best-performing model identified from **500 run** on the Fashion-MNIST dataset, to wandb.

## Requirements
Ensure you have the following installed:
- Python 3.8+
- NumPy
- WandB
- Matplotlib
- Seaborn
- Scikit-learn


To run the script:
```terminal
python code7.py
```

### Expected Outputs
- **Test Accuracy**: Computed on the Fashion-MNIST dataset.
- **Confusion Matrix**: Logged and visualized using `wandb`.

### WandB Logging
- The script initializes a `wandb` run with the best hyperparameters.
- Each epoch logs:
  - Training Loss
  - Training Accuracy
  - Validation Loss
  - Validation Accuracy
- The confusion matrix is logged after training.

## Configuration
The script automatically imports the best hyperparameters from `best_run_config`:
- Number of hidden layers
- Batch size
- Activation function
- Optimizer
- Learning rate
- Weight initialization method
- Weight decay

## Results Interpretation
- The test accuracy gives insight into how well the model generalizes.
- The confusion matrix highlights misclassifications, helping in performance analysis.

## Notes
- Ensure the Fashion-MNIST dataset (`fashion-mnist.npz`) is available in the directory.
- The script runs directly without using a `wandb` sweep.

For any modifications, update `best_run_config.py` to change hyperparameters before running the script.






### Question 8: Comparing Cross-Entropy Loss with Squared Error Loss

`code8.py` compares the effectiveness of cross-entropy loss and squared error loss in training a neural network on the Fashion-MNIST dataset. The experiment is conducted using different optimizer settings in a WandB sweep and logs the results to WandB.


##### Run the script to start a WandB sweep:
   ```terminal
   python code8.py
   ```

#### Conclusion
I did **50 runs** using wandb sweep. The graph logged on wandb clearly shows that both **training and validation loss are less in cross-entropy** compared to **squired error**


### Question 9
Github link 
`https://github.com/amar-at-iitm/da6401_assignment1`




### Question 10: MNIST Experimentation and Training with Various Hyperparameters
From the **500 runs** of `code4.py`, I picked the 3 runs with the highest validation accuracy. From the wandb run **overview** got the configuration parameters and then applied them to **MINIST** dataset using `train.py` and logged the result to **wandb**.


### `train.py` Specification
`train.py` is designed to train a fully connected neural network on the MNIST and Fashion-MNIST datasets using only NumPy. It includes features such as:
- Customizable network architecture
- Multiple optimizer choices
- Configurable training hyperparameters
- Weights & Biases (WandB) integration for experiment tracking


#### Command-line Arguments
The `train.py` script accepts several command-line arguments for training configuration. Example usage:
```terminal
python train.py --wandb_project my_project --wandb_entity my_entity --dataset mnist \
                --epochs 10 --batch_size 32 --loss cross_entropy --optimizer adam \
                --learning_rate 0.01 --num_layers 2 --hidden_size 128 --activation ReLU
```

## Final Thoughts
This assignment explored deep learning on the **Fashion-MNIST** dataset using a **fully connected neural network** implemented from scratch with **NumPy**. Several key aspects were covered, including:

- **Model Training & Hyperparameter Optimization:**
  - Used **WandB sweeps** to identify the best hyperparameters.
  - Implemented multiple optimizers and weight initialization methods.

- **Performance Evaluation:**
  - Compared **cross-entropy loss** with **squared error loss** to analyze their effectiveness.
  - Plotted confusion matrices and accuracy metrics to visualize model performance.

- **Code Modularity & Best Practices:**
  - Separated functionalities into dedicated modules for **forward propagation, backpropagation, optimizers, and configuration management**.
  - Ensured **clean, well-structured code** following best software engineering practices.

- **Reproducibility & Transparency:**
  - Training and testing datasets were properly **split and randomized** to prevent data leakage.
  - The code is **fully documented** and adheres to ethical ML practices (no data cheating).

Overall, this assignment provides a strong foundation for **neural network training, evaluation, and optimization**. Future improvements could include **CNNs for better feature extraction, advanced regularization techniques, or larger-scale hyperparameter tuning**.


### Self Declaration 
I, Amar Kumar, swear on my honour that I have written the code and the report by myself and have not copied it from the internet or other students.
