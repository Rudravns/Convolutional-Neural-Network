import os
import time
import numpy as np
from CNN import CNN
import load_fruits as dataset # Changed from load_mnist to your updated dataset module

# Loss pipeline helpers
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def categorical_cross_entropy(predictions, target):
    predictions = np.clip(predictions, 1e-15, 1.0 - 1e-15)
    return -np.sum(target * np.log(predictions))

# Just to time execution profiles
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
    num_samples_to_train = config.get("num_samples", 2000)
    save_path = config.get("save_path", "fruit_cnn_weights.pkl")
    target_size = config.get("target_size", (32, 32))

    print(f"Loading fruit dataset 'train' split samples...")
    # 1. Load the custom fruit images using your new prefix-matching function
    images, labels = dataset.load_fruit_dataset("train", target_size=target_size)

    # 2. Normalize pixel values from to [0.0, 1.0]
    # No reshape needed! The array is already (Num, 3, 32, 32)
    images = images.astype(np.float32) / 255.0

    # 3. Training Loop Execution
    indices = np.arange(images.shape[0])
    
    # Caps training to your config preference so it doesn't run forever on pure NumPy
    num_samples_to_train = min(num_samples_to_train, len(images))
    print(f"Starting training on {num_samples_to_train} fruit images...")
    
    for epoch in range(config["epochs"]):
        np.random.shuffle(indices)
        total_loss = 0
        correct = 0
        
        for step, idx in enumerate(indices[:num_samples_to_train]):
            X = images[idx]  # Shape: (3, 32, 32)
            
            target = np.zeros(config["output_size"])
            target[labels[idx]] = 1
            
            # Forward pass
            raw_scores = model.forward(X)
            predictions = softmax(raw_scores)
            
            # Metrics
            total_loss += categorical_cross_entropy(predictions, target)
            if np.argmax(predictions) == labels[idx]:
                correct += 1
                
            # Backward and step updates
            loss_grad = predictions - target
            model.backward(loss_grad)
            model.update(config["learning_rate"])
            
            # Print steps every 100 images so you know it's making progress
            if (step + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} | Step {step+1}/{num_samples_to_train}...", end="\r")
                
        acc = (correct / num_samples_to_train) * 100
        print(f"Epoch {epoch+1}/{config['epochs']} | Avg Loss: {total_loss/num_samples_to_train:.4f} | Accuracy: {acc:.2f}%")

    # Save the trained model parameters locally!
    model.save_weights(save_path)
    print(f"Model weights saved successfully to '{save_path}'!")
    
@time_it
def test_model(model: CNN, config: dict):
    target_size = config.get("target_size", (32, 32))
    print(f"\n=============================================")
    print(f"   Evaluating Model on Unseen Test Fruits...  ")
    print(f"=============================================")

    # 1. Load the TEST split of your fruits dataset
    images, labels = dataset.load_fruit_dataset("test", target_size=target_size)

    # Convert to float and scale
    images = images.astype(np.float32) / 255.0

    # Cap evaluation samples so it returns results rapidly
    num_test_samples = config.get("num_test_samples", 500) 
    num_test_samples = min(num_test_samples, len(images))

    correct = 0
    total_loss = 0

    # 2. Evaluation Loop (Forward Pass ONLY)
    print(f"Evaluating across {num_test_samples} unseen validation images...")
    for idx in range(num_test_samples):
        X = images[idx]
        
        target = np.zeros(config["output_size"])
        target[labels[idx]] = 1
        
        # Inference
        raw_scores = model.forward(X)
        predictions = softmax(raw_scores)
        
        # Metrics tracking
        total_loss += categorical_cross_entropy(predictions, target)
        if np.argmax(predictions) == labels[idx]:
            correct += 1

        if (idx + 1) % 100 == 0:
            print(f"  Evaluated {idx + 1}/{num_test_samples} test images...", end="\r")

    acc = (correct / num_test_samples) * 100
    avg_loss = total_loss / num_test_samples
    
    print(f"\nFinal Test Accuracy: {acc:.2f}% | Test Avg Loss: {avg_loss:.4f}\n")
    return acc, avg_loss


if __name__ == "__main__":
    os.system("cls")
    
    # Configuration tailored specifically for 3-Channel RGB 32x32 Fruit Processing
    config = {
        "input_shape": (3, 32, 32),        
        "output_size": 10,                 
        "target_size": (32, 32),           
        "conv_layers": [
            {"filters": 12, "kernel_size": 3}, # Slightly increased filters for more feature detection
            {"filters": 24, "kernel_size": 3}  # Increased filters
        ],
        "pool_size": 2,
        "fc_hidden": (128, 80, 40),     
        "epochs": 5,                       # Give it more loops to find the pattern
        "learning_rate": 0.001,            # Lowered step size for stability
        "num_samples": 6000,               # Way more images so it actually learns
        "num_test_samples": 500,
        "save_path": "fruit_cnn_weights.pkl"
    }

    # Initialize model
    model = CNN(config)
    
    # Train
    train_model(model, config)
    print("Training complete! Moving to test metrics evaluation phase... \n")
    
    # Evaluate
    test_model(model, config)