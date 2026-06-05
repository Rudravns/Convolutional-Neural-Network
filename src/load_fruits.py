import os
import struct
import gzip
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image  # NEW: Pillow library for reading standard images

# Gets the folder containing this script
BASE_DIR = Path(__file__).resolve().parent

# Data Directories
DATA_DIR = BASE_DIR.parent / "mnist-dataset"
FOOD_DATA_DIR = BASE_DIR.parent / "food-dataset"  # NEW: Directory for the Fruit/Food dataset

# Available EMNIST datasets
EMNIST_DATASETS = [
    "emnist-balanced", "emnist-byclass", "emnist-bymerge",
    "emnist-digits", "emnist-letters", "emnist-mnist",
]

# NEW: The 10 explicit classes you want to train on for the food AI
FOOD_CLASSES = [
    "Apple", "Banana", "Orange", "Strawberry", "Kiwi", 
    "Lemon", "Mango", "Plum", "Tomato", "Pineapple"
]


# ======================
# MNIST extraction & loading
# ======================
MNIST_GZ_FILES = [
    "emnist-digits-train-images-idx3-ubyte.gz",
    "emnist-digits-train-labels-idx1-ubyte.gz",
    "emnist-digits-test-images-idx3-ubyte.gz",
    "emnist-digits-test-labels-idx1-ubyte.gz",
]

def extract_mnist():
    for file in MNIST_GZ_FILES:
        gz_path = DATA_DIR / file
        out_path = DATA_DIR / file.replace('.gz', '')

        if os.path.exists(out_path): continue
        if not os.path.exists(gz_path): continue

        with gzip.open(gz_path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted {file}")

def load_mnist_images(filename):
    with open(os.path.join(DATA_DIR, filename), "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
        return np.swapaxes(images, 1, 2)

def load_mnist_labels(filename):
    with open(os.path.join(DATA_DIR, filename), "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def load_mnist_dataset(split: str = "train"):
    if split == "train":
        images = load_mnist_images("emnist-digits-train-images-idx3-ubyte")
        labels = load_mnist_labels("emnist-digits-train-labels-idx1-ubyte")
    elif split == "test":
        images = load_mnist_images("emnist-digits-test-images-idx3-ubyte")
        labels = load_mnist_labels("emnist-digits-test-labels-idx1-ubyte")
    else: raise ValueError("split must be 'train' or 'test'")
    return images, labels

def load_emnist_images(dataset_name: str, split: str = "train"):
    filename = f"emnist-{dataset_name}-{split}-images-idx3-ubyte"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
        return np.swapaxes(images, 1, 2)

def load_emnist_labels(dataset_name: str, split: str = "train"):
    filename = f"emnist-{dataset_name}-{split}-labels-idx1-ubyte"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def load_emnist_dataset(dataset_name: str, split: str = "train"):
    images = load_emnist_images(dataset_name, split)
    labels = load_emnist_labels(dataset_name, split)
    return images, labels



def load_fruit_dataset(split="train", target_size=(32, 32)):
    """
    Scans the 'food-dataset/train' or 'food-dataset/test' folder for your 10 target fruit categories.
    Handles sub-variety folders (e.g., 'Apple 5', 'Apple Red') by matching the prefix.
    Loads the RGB images, resizes them down to 32x32, and shapes them for the CNN.
    """
    split_dir = FOOD_DATA_DIR / split
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Directory missing: {split_dir}. Make sure you extracted the folders!")

    print(f"Scanning {split_dir} for 10 fruit classes...")
    
    images = []
    labels = []

    # Get a list of all actual folders inside the train or test directory
    all_folders = [f for f in os.listdir(split_dir) if os.path.isdir(split_dir / f)]

    # Loop through each of your 10 target classes
    for label_idx, class_name in enumerate(FOOD_CLASSES):
        # Find every folder that starts with your class name (e.g., "Apple 5", "Apple Red")
        matching_folders = [f for f in all_folders if f.lower().startswith(class_name.lower())]
        
        if not matching_folders:
            print(f"  [!] No folders found starting with: {class_name}")
            continue
            
        file_count = 0
        # Process each matching folder group together under the same label index
        for folder in matching_folders:
            class_dir = split_dir / folder
            
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = class_dir / file_name
                    try:
                        # 1. Load image, force RGB, and resize to (32, 32)
                        img = Image.open(img_path).convert('RGB').resize(target_size)
                        
                        # 2. Convert to Numpy Array: Shape becomes (32, 32, 3)
                        img_array = np.array(img, dtype=np.uint8)
                        
                        # 3. Transpose axes from (Height, Width, Channels) to (Channels, Height, Width)
                        img_array = np.transpose(img_array, (2, 0, 1))
                        
                        images.append(img_array)
                        labels.append(label_idx)  # They all get grouped under the same label!
                        file_count += 1
                    except Exception:
                        pass
                        
        print(f"  -> Grouped & loaded {file_count} total images for prefix: {class_name}")

    if len(images) == 0:
        raise ValueError("No images were loaded! Check your folder structure and names.")

    # Return as unified numpy arrays
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.uint8)


# ======================
# Debug / test
# ======================
# ======================
# Debug / test
# ======================
if __name__ == "__main__":
    os.system("cls")
    print("=== Testing Data Loaders ===")
    
    try:
        # Load the dataset
        fruit_imgs, fruit_lbls = load_fruit_dataset("train")
        print(f"\nSUCCESS! Loaded total dataset shape: {fruit_imgs.shape}")
        print(f"Shape of just ONE image: {fruit_imgs[0].shape} (Channels, Height, Width)")
        
        # Plot the first fruit to make sure it looks right!
        plt.figure(figsize=(3, 3))
        
        # FIX: Explicitly grab the FIRST image before transposing
        first_image = fruit_imgs[0]
        display_img = np.transpose(first_image, (1, 2, 0))
        
        plt.imshow(display_img)
        
        # FIX: Explicitly grab the FIRST label to look up the text name
        first_label_idx = fruit_lbls[0]
        plt.title(f"Class: {FOOD_CLASSES[first_label_idx]}")
        
        plt.axis('off')
        plt.show()
    except FileNotFoundError as e:
        print(f"\n{e}")