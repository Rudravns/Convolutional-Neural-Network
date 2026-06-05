# pyright: reportArgumentType = false
from pathlib import Path

import pygame
import CNN
import numpy as np
import os
import train_fruits  # Changed to look at your fruit training pipeline helper

# Define the exact 10 fruit categories in index order matching your loader
FOOD_CLASSES = [
    "Apple", "Banana", "Orange", "Strawberry", "Kiwi", 
    "Lemon", "Mango", "Plum", "Tomato", "Pineapple"
]

class test:
    def __init__(self):
        os.system("cls")
        pygame.init()

        # Window settings
        self.screen = pygame.display.set_mode((1100, 840))
        pygame.display.set_caption("Scratch CNN - Fruit Variety Classifier")
        self.clock = pygame.time.Clock()
        self.dt = 0

        # Map initialization (32x32 pixels, scaled visually to fill space)
        self.map = MAP(32, 32, cell_size = 20)

        self.font = pygame.font.SysFont("Consolas", 24)
        self.predictions = np.zeros(10)
        
        # Configuration tailored specifically for 3-Channel RGB 32x32 Fruit Processing
        cnn_config = {
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
        
        self.model = CNN.CNN(cnn_config)
        try:
            # Load the matching fruit matrix weights
            self.model.load_weights("fruit_cnn_weights.pkl")
            print("Successfully loaded pre-trained fruit model weights!")
        except FileNotFoundError:
            print("[-] No weights found. Ensure you run 'python train_fruits.py' first!")

        self.img = 0

    def predict_drawing(self):
        """Passes the current canvas array into the network for inference."""
        canvas_grid = self.map.map  # Shape: (3, 32, 32)
        
        if np.sum(canvas_grid) == 0:
            return
            
        # Forward prediction pass through our pure-numpy architecture
        raw_scores = self.model.forward(canvas_grid)
        self.predictions = train_fruits.softmax(raw_scores).flatten()
        
        predicted_idx = np.argmax(self.predictions)
        confidence = self.predictions[predicted_idx] * 100
        
        print(f"Prediction: {FOOD_CLASSES[predicted_idx]} | Confidence: {confidence:.2f}%", end="\r")

    def draw_ui(self):
        """Renders the fruit class prediction text list on the right sidebar."""
        start_x = 750
        start_y = 80
        
        header = self.font.render("CNN Predictions:", True, (0, 0, 0))
        self.screen.blit(header, (start_x, 40))

        # Loop through each of the 10 categories
        for i in range(10):
            prob = self.predictions[i]
            
            # Highlight the highest scoring category in green, others in dark grey
            if i == np.argmax(self.predictions) and np.sum(self.predictions) > 0:
                color = (0, 150, 0)
            else:
                color = (60, 60, 60)
            
            # Formats string to look like: "Apple:  45.21%"
            text_str = f"{FOOD_CLASSES[i]:12s}: {prob*100:5.2f}%"
            text_surface = self.font.render(text_str, True, color)
            
            self.screen.blit(text_surface, (start_x, start_y + (i * 45)))

    def run(self):
        while True:
            # A clean, light-colored background looks better behind vibrant fruit colors
            self.screen.fill((240, 240, 240))
            self.dt = self.clock.tick(60) / 1000
            
            # Render our downscaled low-res canvas matrix and prediction metrics text
            self.map.draw(topright = (50, 50))
            self.draw_ui()

            # Event Loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()
                    if event.key == pygame.K_c:
                        self.map.reset()
                        self.predictions = np.zeros(10)
                    
                    # Press 'L' to feed an external image file directly into the model
                    if event.key == pygame.K_l:
                        # Checks for common naming formats dynamically
                        target_images = ["apple.jpeg", "banana.jpeg", "apple.png", "banana.png", "kivi.png", "kivi.jpg"]  # Extend this list with more expected filenames as needed
                        img_found = False

                        # Absolute path to the directory where main.py sits, then up one level, then into img
                        img_dir = Path(__file__).resolve().parent.parent / "imgs"
                        img_name = str(img_dir / target_images[self.img])
                        print(f"\nAttempting to load image: {img_name} ...")

                        if os.path.exists(img_name):
                            try:
                                self.map.set(img_name)
                                self.predict_drawing()
                                print(f"\nLoaded '{img_name}' successfully into CNN input matrix!")
                                img_found = True
                                self.img = (self.img + 1) % len(target_images)
                                break
                            except Exception as e:
                                print(f"\nError processing image: {e}")
                        
                        if not img_found:
                            print("\n[!] No test image found")
                        
            pygame.display.flip()


class MAP:
    def __init__(self, row, col, cell_size = 30):
        self.row = row
        self.col = col
        self.cell_size = cell_size
        self.map = np.zeros((3, row, col))
        self.screen = pygame.display.get_surface()
    def set(self, path):
        # 1. Load the original image with transparency
        loaded_img = pygame.image.load(path).convert_alpha()
        
        # 2. AUTOMATIC CROP: Find the tightest rectangle around the actual fruit pixels
        # (Strips away any massive empty space at the bottom/sides of the PNG)
        fruit_rect = loaded_img.get_bounding_rect()
        
        tight_fruit = pygame.Surface((fruit_rect.width, fruit_rect.height), pygame.SRCALPHA)
        tight_fruit.blit(loaded_img, (0, 0), fruit_rect)
        
        img_w, img_h = tight_fruit.get_size()
        
        # 3. Determine the size of the square canvas (pick the larger side)
        square_size = max(img_w, img_h)
        
        # 4. Create a temporary pure white square surface (matching the dataset)
        square_bg = pygame.Surface((square_size, square_size))
        square_bg.fill((255, 255, 255))
        
        # 5. Calculate perfect centering offsets for the cropped fruit
        offset_x = (square_size - img_w) // 2
        offset_y = (square_size - img_h) // 2
        
        # 6. Paste the tightly cropped fruit directly into the center
        square_bg.blit(tight_fruit, (offset_x, offset_y))
        
        # 7. Smoothly downscale the perfect, centered square canvas to exactly 32x32
        small_img = pygame.transform.smoothscale(square_bg, (self.col, self.row))
        self.map = np.zeros((3, self.row, self.col))
        
        for r in range(self.row):
            for c in range(self.col):
                color = small_img.get_at((c, r))
                
                # Reshape into channel-first orientation [Channels, Height, Width] normalized to [0.0, 1.0]
                self.map[0][r][c] = color.r / 255.0
                self.map[1][r][c] = color.g / 255.0
                self.map[2][r][c] = color.b / 255.0

    def get(self, row, col):
        return self.map[:, row, col]

    def draw(self, topright = (0, 0)):
        for r in range(self.row):
            for c in range(self.col):
                # FIX: Explicitly index channel 0 (R), 1 (G), and 2 (B)
                rgb = (
                    int(self.map[0][r][c] * 255), 
                    int(self.map[1][r][c] * 255), 
                    int(self.map[2][r][c] * 255)
                )
                rect = (topright[0] + c * self.cell_size, topright[1] + r * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, rgb, rect)

    def reset(self):
        self.map = np.zeros((3, self.row, self.col))
        print("\nCanvas Cleared.                                                 ", end="\r")


if __name__ == "__main__":
    app = test()
    app.run()