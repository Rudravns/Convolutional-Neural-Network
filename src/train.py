import os
import time

import numpy as np
from CNN import CNN
import dataset  # Using your custom dataset module

# Loss pipeline helpers
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def categorical_cross_entropy(predictions, target):
    predictions = np.clip(predictions, 1e-15, 1.0 - 1e-15)
    return -np.sum(target * np.log(predictions))


#just to time it
def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Train the model
@time_it
def train_model(model: CNN, config: dict):
    # Configuration-driven parameters
    num_samples_to_train = config.get("num_samples", 2000)
    save_path = config.get("save_path", "mnist_cnn_weights.pkl")
    dataset_name = config.get("dataset_name", "digits")

    # 1. Extract and load dataset using your file logic
    print(f"Extracting archive files for '{dataset_name}' if needed...")
    dataset.extract_mnist()

    print(f"Loading '{dataset_name}' dataset samples...")
    # Using the flexible EMNIST loader
    images, labels = dataset.load_emnist_dataset(dataset_name, "train")

    # Reshape from (Num, 28, 28) to (Num, 1, 28, 28) and scale pixels to [0.0, 1.0]
    images = images.reshape(-1, 1, 28, 28) / 255.0

    # 2. Training Loop Execution
    indices = np.arange(images.shape[0])
    print(f"Starting training on {num_samples_to_train} images...")
    for epoch in range(config["epochs"]):
        np.random.shuffle(indices)
        total_loss = 0
        correct = 0
        
        for step, idx in enumerate(indices[:num_samples_to_train]):
            X = images[idx]
            
            target = np.zeros(config["output_size"])
            target[labels[idx]] = 1
            
            # Forward pass
            raw_scores = model.forward(X)
            predictions = softmax(raw_scores)
            
            # Metrics
            total_loss += categorical_cross_entropy(predictions, target)
            if np.argmax(predictions) == labels[idx]:
                correct += 1
                
            # Backward and step
            loss_grad = predictions - target
            model.backward(loss_grad)
            model.update(config["learning_rate"])
            
        acc = (correct / num_samples_to_train) * 100
        print(f"Epoch {epoch+1}/{config['epochs']} | Avg Loss: {total_loss/num_samples_to_train:.4f} | Accuracy: {acc:.2f}%")

    # Save the trained model parameters locally!
    model.save_weights(save_path)
    
@time_it
def test_model(model: CNN, config: dict):
    dataset_name = config.get("dataset_name", "digits")
    print(f"\n=============================================")
    print(f"  Evaluating Model on Unseen Test Data...  ")
    print(f"=============================================")

    # 1. Load the TEST split of the dataset
    images, labels = dataset.load_emnist_dataset(dataset_name, "test")

    # Reshape and normalize (exactly the same as training)
    images = images.reshape(-1, 1, 28, 28) / 255.0

    # For a custom NumPy CNN, testing all 10k+ images might take a few minutes. 
    # Let's cap it to a specific number (or test the whole thing if you want!)
    num_test_samples = config.get("num_test_samples", 1000) 
    num_test_samples = min(num_test_samples, len(images))

    correct = 0
    total_loss = 0

    # 2. Evaluation Loop (Forward Pass ONLY - No backprop or weight updates!)
    for idx in range(num_test_samples):
        X = images[idx]
        
        target = np.zeros(config["output_size"])
        target[labels[idx]] = 1
        
        # Forward pass
        raw_scores = model.forward(X)
        predictions = softmax(raw_scores)
        
        # Metrics tracking
        total_loss += categorical_cross_entropy(predictions, target)
        if np.argmax(predictions) == labels[idx]:
            correct += 1

        # Optional: Print progress every 200 images so you know it's not frozen
        if (idx + 1) % 200 == 0:
            print(f"Evaluated {idx + 1}/{num_test_samples} images...", end="\r")

    acc = (correct / num_test_samples) * 100
    avg_loss = total_loss / num_test_samples
    
    print(f"\nFinal Test Accuracy: {acc:.2f}% | Test Avg Loss: {avg_loss:.4f}\n")
    return acc, avg_loss




if __name__ == "__main__":
    # 1. Config matching your spatial structure
    os.system("cls")
    config = {
        "input_shape": (1, 28, 28),
        "output_size": 10, # Digits 0-9 require 10 classes
        "conv_layers": [
            {"filters": 8, "kernel_size": 3},
            {"filters": 16, "kernel_size": 3}
        ],
        "pool_size": 2,
        "fc_hidden": (128, 80, 40),
        "epochs": 5,
        "learning_rate": 0.005,
        "num_samples": 2000,
        "dataset_name": "digits"
    }

    # Initialize model
    model = CNN(config)
    train_model(model, config)
    print("Training complete! Now evaluating on test set... \n")
    test_model(model, config)