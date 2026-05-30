#pyright:  reportArgumentType = false
import pygame, CNN
import numpy as np
import os
import train

class test:
    def __init__(self):
        #init
        os.system("cls")
        pygame.init()

        #window
        self.screen = pygame.display.set_mode((840, 840))
        self.clock = pygame.time.Clock()
        self.dt = 0

        #map
        self.map = MAP(28, 28, cell_size = 30)

        #config
        cnn_config = {
            "input_shape": (1, 28, 28),     # Channels, Height, Width
            "output_size": 10,              # Digits 0-9
            "conv_layers": [
                {"filters": 8, "kernel_size": 3},
                {"filters": 16, "kernel_size": 3}
            ],
            "pool_size": 2,                 # For Max Pooling
            "fc_hidden": (128, 80, 40),     # Must match train.py exactly!
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.01           # Often needs to be lower for CNNs
        }
        self.model = CNN.CNN(cnn_config)
        try:
            self.model.load_weights("mnist_cnn_weights.pkl")
            print("Successfully loaded pre-trained CNN weights!")
        except FileNotFoundError:
            print("No pre-trained weights found. Starting with random initialization.")

    def predict_drawing(self):
        """Passes the current canvas array into the network for inference."""
        # Get the 2D canvas array (28, 28)
        canvas_grid = self.map.map
        
        # Check if canvas is entirely empty to avoid predicting on noise
        if np.sum(canvas_grid) == 0:
            return
            
        # Format shape to match your CNN's input needs: (1, 28, 28)
        input_data = canvas_grid.reshape(1, 28, 28)
        
        # Run forward prediction pass
        raw_scores = self.model.forward(input_data)
        probabilities = train.softmax(raw_scores).flatten()
        
        predicted_digit = np.argmax(probabilities)
        confidence = probabilities[predicted_digit] * 100
        
        # Print a clean live updates console statement
        print(f"Predicted Digit: {predicted_digit} | Confidence: {(confidence):.2f}%", end="\r")      

    def run(self):
        while True:
            #reset screen
            self.screen.fill((255, 100, 100))
            self.dt = self.clock.tick(60) / 1000

            #map and mouse update
            self.update_mouse()
            self.map.draw(topright = (0, 0))
            
            #event loop
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

            pygame.display.update() #update screen
    
    def update_mouse(self):
        """
        Left click to draw, right click to erase.
        Runs predictions only when the map actively changes.
        """
        mouse_pos = pygame.mouse.get_pos()
        col = mouse_pos[0] // self.map.cell_size
        row = mouse_pos[1] // self.map.cell_size

        # FIX 1: Add a safety guard boundary condition to prevent IndexError crashes
        if 0 <= row < self.map.row and 0 <= col < self.map.col:
            buttons = pygame.mouse.get_pressed()
            if buttons[0]: # Left click
                if self.map.get(row, col) != 1:  # Only predict if drawing changes a pixel
                    self.map.set(row, col, 1)
                    self.predict_drawing()
            elif buttons[2]: # Right click
                if self.map.get(row, col) != 0:  # Only predict if erasing changes a pixel
                    self.map.set(row, col, 0)
                    self.predict_drawing()
            

class MAP:
    def __init__(self, row, col, cell_size = 30):
        self.row = row
        self.col = col
        self.cell_size = cell_size
        self.map = np.zeros((row, col))
        self.screen = pygame.display.get_surface()
    
    def set(self, row, col, value):
        self.map[row][col] = value

    def get(self, row, col):
        return self.map[row][col]

    def reset(self):
        self.map = np.zeros((self.row, self.col))
        print("\nCanvas Cleared.                                                 ", end="\r")

    def draw(self, topright = (0,0)):
        for i in range(self.row):
            for j in range(self.col):
                if self.map[i][j] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), (topright[0] + j * self.cell_size, topright[1] + i * self.cell_size, self.cell_size, self.cell_size)) 
                pygame.draw.rect(self.screen, (0, 0, 0), (topright[0] + j * self.cell_size, topright[1] + i * self.cell_size, self.cell_size, self.cell_size), 1)    


if __name__ == "__main__":
    app = test()
    app.run()