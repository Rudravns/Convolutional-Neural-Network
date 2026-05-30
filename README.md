# Custom Convolutional Neural Network (CNN) from Scratch

A modular, framework-free 2D Convolutional Neural Network built completely from scratch using Python and **NumPy**. This repository features a fully functional training and evaluation pipeline tested on the EMNIST/MNIST dataset, paired with an interactive, live-inference **PyGame** canvas for real-time digit recognition.

---

## 🚀 Key Features

- **Pure NumPy Implementation:** No dependency on PyTorch, TensorFlow, or Keras. Every matrix operation, activation function, and gradient tracking mechanism is built using standard NumPy operations.
- **Modular Architecture:** Layer types are encapsulated as independent classes (`Conv2D`, `ReLU`, `MaxPool2D`, `Flatten`, `Dense`) with unified `forward()` and `backward()` APIs, mimicking real-world deep learning frameworks.
- **Dynamic Structural Math:** Automatically tracks, passes, and flattens variable-sized multi-channel spatial data tensors down to standard fully connected classifier inputs using zero-value mock propagation.
- **Live PyGame Inference Canvas:** Draws $28 \times 28$ high-resolution input cells dynamically, processing real-time feedforward passes continuously without interrupting game loop frame states or crashing bounds.
- **Save/Load Capabilities:** Leverages Python's binary serialization framework (`pickle`) to extract, serialize, and store optimized trained network layer weights to static checkpoint disk files (`.pkl`).

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rudravns/Convolutional-Neural-Network.git
   cd "Convolutional Neural Network"
   ```

2. **Install Dependencies:**
   This project requires only standard scientific Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Setup:**
   The `dataset.py` script automatically handles the extraction of EMNIST/MNIST files. Ensure you have the `.gz` archive files in the `mnist-dataset/` directory relative to the project root.

---

## 💻 Commands & Code API Reference

This section outlines both terminal-level execution scripts and python-level programmatic function calls within the architecture components.

### 1. Terminal Execution Commands

Run these core scripts directly from your terminal workspace inside the project root folder:

* **Install Dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

* **Verify Dataset Archives Standalone:**
  Parses binary IDX content, lists local files, and renders structural matrix shape counts for debug validation.
  ```bash
  python src/dataset.py
  ```

* **Execute Model Training Pipeline:**
  Initializes the weights, iterates backward passes over the specified sample batches, and outputs serialized matrix variables to disk.
  ```bash
  python src/train.py
  ```

* **Launch Interface App Client:**
  Initializes the window layout canvas, loads local checkpoints, and computes feedforward sweeps from real-time mouse movements.
  ```bash
  python src/main.py
  ```

### 2. Python Code API & Function Reference
If you are expanding the network or importing its framework modules into separate files, use the following code patterns and method invocations:

#### A. Data Pipeline Ingestion (src/dataset.py)
```python
import src.dataset as dataset

# 1. Unzip compressed IDX format archives in your local path
dataset.extract_mnist()

# 2. Extract multi-dimensional numpy arrays (automatically transposes EMNIST upright)
# Split options: 'train' or 'test'
images, labels = dataset.load_emnist_dataset(dataset_name="digits", split="train")

print(images.shape)  # Returns: (Num_Samples, 28, 28)
print(labels.shape)  # Returns: (Num_Samples,)

# 3. Quick structural lookup to scan sample volumes stored locally
dataset.list_available_emnist_datasets()
```

#### B. Core Network Control & Custom Layers (src/CNN.py)
```python
from src.CNN import CNN
import numpy as np

# 1. Define your dynamic layer dimensions
cnn_config = {
    "input_shape": (1, 28, 28),
    "output_size": 10,
    "conv_layers": [
        {"filters": 8, "kernel_size": 3},
        {"filters": 16, "kernel_size": 3}
    ],
    "pool_size": 2,
    "fc_hidden": (128, 80, 40)
}

# 2. Instantiate your neural network object
model = CNN(cnn_config)

# 3. Execute Forward Pass (Pass a grayscale image matrix shaped as: Channels, Height, Width)
mock_image = np.random.randn(1, 28, 28)
raw_logits = model.forward(mock_image)  # Returns unnormalized raw scores (Shape: 10,)

# 4. Execute Backward Pass (Pass the evaluated loss gradient array downstream)
# For Categorical Cross-Entropy coupled with Softmax, this gradient shortcut is: Predictions - Target
loss_gradient = np.random.randn(10)
model.backward(loss_gradient)

# 5. Gradient Descent Update (Scales local weight derivatives and deducts them)
learning_rate = 0.005
model.update(learning_rate)

# 6. Weight Persistence Checkpointing
model.save_weights("mnist_cnn_weights.pkl") # Saves internal arrays via binary serialization
model.load_weights("mnist_cnn_weights.pkl") # Re-populates weights back into active memory instances
```

#### C. Training Logic Pipelines & Metrics Helpers (src/train.py)
```python
import src.train as train
from src.CNN import CNN

model = CNN(cnn_config)

# 1. Pipeline Normalization & Optimization Execution Loop
# Runs feature training sweeps, calculates SGD steps, and builds checkpoint files
train.train_model(model, cnn_config)

# 2. Pipeline Test Generalization Valuation Evaluation Loop
# Evaluates accuracy performance scores over unseen evaluation validation testing sets
test_accuracy = train.test_model(model, cnn_config)

# 3. Isolated Activation and Optimization Helpers
probabilities = train.softmax(raw_logits)
loss_value = train.categorical_cross_entropy(probabilities, standard_target_array)
```

---

## 🏃 Usage

### 1. Training the Model
Run the training script to optimize weights on the EMNIST digits dataset. By default, it will train for 5 epochs and save the results to `mnist_cnn_weights.pkl`.
```bash
python src/train.py
```

### 2. Interactive Inference (GUI)
Launch the PyGame window to draw digits and see real-time predictions from your trained model.
```bash
python src/main.py
```

---

## 🎹 GUI Controls

| Key/Action | Description |
| :--- | :--- |
| **Left Click** | Draw on the 28x28 canvas |
| **Right Click** | Erase pixels |
| **'C' Key** | Clear the canvas and reset predictions |
| **ESC Key** | Exit the application |

---

## 📁 Project Structure

- **`src/CNN.py`**: The core engine containing the `CNN` class and individual layer implementations (`Conv2D`, `Dense`, etc.).
- **`src/train.py`**: The training pipeline, including Loss functions (Cross-Entropy) and the Softmax activation.
- **`src/main.py`**: The PyGame-based drawing application and UI logic.
- **`src/dataset.py`**: Utilities for binary idx-file parsing and EMNIST data management.
- **`mnist_cnn_weights.pkl`**: Serialized weight file generated after training.

---

## 📝 Mathematical Implementation Details

- **Optimization:** Stochastic Gradient Descent (SGD).
- **Activation:** ReLU (Hidden Layers) and Softmax (Output Layer).
- **Loss:** Categorical Cross-Entropy.
- **Initialization:** He-style random initialization scaled by 0.1 for stability.