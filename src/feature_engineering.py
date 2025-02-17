import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from skimage.feature import hog

processed_image_array_dir = Path("/Users/harishprabhu/Desktop/Machine_Vision/Machine-Vision/data/processed/")

def load_image_array(category, num_images = 5):
    """Function loads the .npy file"""

    file_path = processed_image_array_dir / f"{category}_train.npy"
    if not file_path.exists():
        print(f"File Not Found at {file_path}. Run Data Ingestion First.")
        return

    print(f"File Found! Loading...\n")

    images = np.load(file_path)
    return images[: num_images]

# Exploring Traditional Methods for Feature Extraction
# -- (A) HISTOGRAM OF PIXEL INTENSITIES --
def plot_histogram(images):
    """plots histogram for colors mentioned in a list."""
    color_channel = ['r', 'g', 'b']
    plt.figure(figsize = (10, 5))
    for i, col in enumerate(color_channel):
        hist = cv2.calcHist([images], [i], None, [256], [0, 256]) # [i] = channel - RGB (3-channel)
        plt.plot(hist, color = col)
        plt.xlim([0, 256]) # defining a range for x - axis
    plt.title("Histogram for RGB")
    plt.show()

# -- (B) Edge Detection --
def plot_edge_detection(images):
    """Functions applies Canny Edge Detection"""
    # Covert to grayscale
    gray = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)

    # Applies Canny edge detection
    # (100, 200) - min max values for low contrast images
    # Make sure you specify (L2Gradient) else: it shows ASSERTION ERROR
    edges = cv2.Canny(gray, 80, 200, 3, L2gradient = True) 
    
    plt.figure(figsize = (8, 8))
    plt.subplot(121)
    plt.imshow(images, cmap = 'gray')
    plt.title("Original Image in gray")
    plt.subplot(122)
    plt.imshow(edges, cmap = "gray")
    plt.title("Canny Edge Detection")
    plt.axis("off")
    plt.show()






def main():
    images = load_image_array("bottle")
    if images is not None:
        for img in images:
            # plot_histogram(img)
            plot_edge_detection(img)
        
        return

    print("Images Failed to load.\n")





if __name__ == "__main__":
    main()