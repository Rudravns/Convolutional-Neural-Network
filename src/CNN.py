import os
import pickle
import numpy as np

class CNN:
    def __init__(self, config):
        in_c, _, _ = config["input_shape"]
        self.layers = []

        # 1. Build Conv and Pooling Blocks dynamically
        current_channels = in_c
        for conv_cfg in config["conv_layers"]:
            self.layers.append(Conv2D(current_channels, conv_cfg["filters"], conv_cfg["kernel_size"]))
            self.layers.append(ReLU())
            self.layers.append(MaxPool2D(config["pool_size"]))
            current_channels = conv_cfg["filters"]

        # 2. Automatically calculate the flattened size for the Dense layers
        dummy_x = np.zeros(config["input_shape"])
        for layer in self.layers:
            dummy_x = layer.forward(dummy_x)
        flattened_size = dummy_x.size

        # 3. Build Classifier Blocks (Flatten -> Dense layers)
        self.layers.append(Flatten())
        
        input_dim = flattened_size
        for hidden_dim in config["fc_hidden"]:
            self.layers.append(Dense(input_dim, hidden_dim))
            self.layers.append(ReLU())
            input_dim = hidden_dim
            
        # Final classification layer
        self.layers.append(Dense(input_dim, config["output_size"]))

    def save_weights(self, filepath="model_weights.pkl"):
        """Saves the weights and biases of all trainable layers."""
        weights_to_save = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                weights_to_save.append({
                    'weights': layer.weights,
                    'biases': layer.biases
                })
        
        with open(filepath, 'wb') as f:
            pickle.dump(weights_to_save, f)
        print(f"Successfully saved trained weights to {filepath}!")

    def load_weights(self, filepath="model_weights.pkl"):
        """Loads weights and biases back into the trainable layers."""
        with open(filepath, 'rb') as f:
            saved_weights = pickle.load(f)

        # Filter for layers that actually have trainable parameters
        trainable_layers = [l for l in self.layers if hasattr(l, 'weights')]

        if len(trainable_layers) != len(saved_weights):
            raise ValueError(f"Architecture mismatch! Model has {len(trainable_layers)} trainable layers, but file has {len(saved_weights)}.")

        for i, layer in enumerate(trainable_layers):
            if layer.weights.shape != saved_weights[i]['weights'].shape:
                raise ValueError(f"Shape mismatch in layer {i}! Expected {layer.weights.shape}, got {saved_weights[i]['weights'].shape}. "
                                 "Ensure your config matches the one used during training.")
            
            layer.weights = saved_weights[i]['weights']
            layer.biases = saved_weights[i]['biases']
        print(f"Successfully loaded weights from {filepath}!")

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

            
    def update(self, learning_rate):
        """Update weights and biases for all layers that have them."""
        for layer in self.layers:
            if hasattr(layer, 'weights'): 
                layer.update_weights(learning_rate)


class Conv2D:
    def __init__(self, in_channels, num_filters, kernel_size):
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        
        # Initialize filters (weights) and biases
        self.weights = np.random.randn(num_filters, in_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros(num_filters)

    def forward(self, input_data):
        self.input_data = input_data  # Saved for backpropagation
        in_c, in_h, in_w = input_data.shape
        
        out_h = in_h - self.kernel_size + 1
        out_w = in_w - self.kernel_size + 1
        output = np.zeros((self.num_filters, out_h, out_w))
        
        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    input_slice = input_data[:, i:i+self.kernel_size, j:j+self.kernel_size]
                    output[f, i, j] = np.sum(input_slice * self.weights[f]) + self.biases[f]
                    
        return output

    def backward(self, output_gradient):
        # 1. Initialize empty arrays matching the shapes of our parameters and input data
        self.dW = np.zeros_like(self.weights) # A gradient array for each filter's weights
        self.dB = np.zeros_like(self.biases)
        input_gradient = np.zeros_like(self.input_data)
        
        num_filters, out_h, out_w = output_gradient.shape
        
        # 2. Slide back through the exact same positions as the forward pass
        for f in range(num_filters):
            # Bias gradient is the sum of the entire incoming gradient map for this filter
            self.dB[f] = np.sum(output_gradient[f])
            
            for i in range(out_h):
                for j in range(out_w):
                    # Grab the exact same input slice that generated this specific output pixel
                    input_slice = self.input_data[:, i:i+self.kernel_size, j:j+self.kernel_size]
                    
                    # Accumulate filter gradients: slice multiplied by the gradient at this position
                    self.dW[f] += input_slice * output_gradient[f, i, j]
                    
                    # Accumulate input gradients: filter weights multiplied by the gradient at this position
                    input_gradient[:, i:i+self.kernel_size, j:j+self.kernel_size] += self.weights[f] * output_gradient[f, i, j]
                    
        return input_gradient

    def update_weights(self, lr):
        # Stochastic Gradient Descent parameter updates
        self.weights -= lr * self.dW
        self.biases -= lr * self.dB

class ReLU:
    def __init__(self):
        pass

    def forward(self, input_data):
        self.input_data = input_data  # Save for backpropagation later
        # np.maximum compares every element in the array against 0
        return np.maximum(0, input_data)
    
    def backward(self, output_gradient):
        # Multiply element-wise: passes gradient through only where input_data > 0
        return output_gradient * (self.input_data > 0)

class MaxPool2D:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, input_data):
        self.input_data = input_data  # Save for backpropagation later
        num_channels, in_h, in_w = input_data.shape
        
        # A pool_size of 2 with stride 2 cuts dimensions in half
        out_h = in_h // self.pool_size
        out_w = in_w // self.pool_size
        
        output = np.zeros((num_channels, out_h, out_w))
        
        for c in range(num_channels):
            for i in range(out_h):
                for j in range(out_w):
                    # Calculate the starting row and column indices in the input image
                    start_r = i * self.pool_size
                    start_c = j * self.pool_size
                    
                    # Extract the 2x2 pool slice
                    pool_slice = input_data[c, start_r:start_r+self.pool_size, start_c:start_c+self.pool_size]
                    
                    # Store only the maximum value found in that 2x2 slice
                    output[c, i, j] = np.max(pool_slice) # channel c, output row i, output column j
                    
        return output
    
    def backward(self, output_gradient):
        # Initialize an array of zeros matching the original large shape (e.g., 8, 26, 26)
        input_gradient = np.zeros_like(self.input_data)
        num_channels, out_h, out_w = output_gradient.shape
        
        for c in range(num_channels):
            for i in range(out_h):
                for j in range(out_w):
                    start_r = i * self.pool_size
                    start_c = j * self.pool_size
                    
                    # 1. Grab the original 2x2 slice from the forward pass input
                    pool_slice = self.input_data[c, start_r:start_r+self.pool_size, start_c:start_c+self.pool_size]
                    
                    # 2. Create a boolean mask where True marks the maximum element's location
                    mask = (pool_slice == np.max(pool_slice))
                    
                    # 3. Route the small incoming gradient value into the max location of our large array
                    # (Multiplying by mask keeps non-max locations as 0)
                    input_gradient[c, start_r:start_r+self.pool_size, start_c:start_c+self.pool_size] += mask * output_gradient[c, i, j]
                    
        return input_gradient

class Flatten:
    def forward(self, input_data):
        self.input_shape = input_data.shape  # Save for backpropagation later
        return input_data.flatten()  # Convert to 1D array
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)  # Reshape back to original dimensions

class Dense:
    def __init__(self, input_size, output_size):
        # Standard FNN weight initialization
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input_data = input_data  # Save for backpropagation later
        
        # Standard FNN forward pass: Y = XW + B
        # (Using dot product so it handles vectors or batches smoothly)
        return np.dot(input_data, self.weights) + self.biases
   
    def backward(self, output_gradient):
        # output_gradient comes from the layer ahead of this one
        
        # 1. Calculate gradients w.r.t weights and biases
        # np.outer multiplies every element of input_data by every element of output_gradient
        self.dW = np.outer(self.input_data, output_gradient)
        self.dB = output_gradient
        
        # 2. Calculate gradient w.r.t input to pass to the previous layer
        input_gradient = np.dot(output_gradient, self.weights.T)
        return input_gradient

    def update_weights(self, lr):
        # This is what your CNN.update() loop calls!
        self.weights -= lr * self.dW
        self.biases -= lr * self.dB


if __name__ == "__main__":
    # Simulate a 28x28 single-channel (grayscale) image like MNIST
    os.system("cls")
    # 1. Start with a mock grayscale image (1 channel, 28x28)
    img = np.random.randn(1, 28, 28)
    model = CNN(config={
        "input_shape": (1, 28, 28),
        "conv_layers": [
            {"filters": 8, "kernel_size": 3},
            {"filters": 16, "kernel_size": 3}
        ],
        "pool_size": 2,
        "fc_hidden": (64, 32),
        "output_size": 10
    })
    

    print("Input shape:", img.shape)
    output = model.forward(img)
    print("Output shape:", output.shape)