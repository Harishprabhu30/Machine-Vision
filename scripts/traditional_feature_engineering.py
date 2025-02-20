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

# -- (C) Histogram of Oriented Gradients (HoG)
def plot_Gradient_Hist(images):
    """Function extracts HoG features and displays them"""
    gray = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)
    features, hog_images = hog(gray, pixels_per_cell = (8, 8), cells_per_block = (2, 2), visualize = True, block_norm = "L2-Hys")

    plt.figure(figsize = (8, 8))
    plt.subplot(121)
    plt.imshow(images, cmap = "gray")
    plt.title("Original Image in Gray")
    plt.subplot(122)
    plt.imshow(hog_images, cmap = "gray")
    plt.title("HoG Image")
    plt.axis("off")
    plt.show()
    return features

###########################################################
# Keypoint Based Feature Extraction (SIFT, ORB)
# -- (A) SIFT --
def plot_sift(images):
    """Function detects keypoinys and descritptor"""
    gray = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create() # Initializze sift detector
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    img_sift = cv2.drawKeypoints(images, keypoints, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(figsize = (5, 5))
    plt.subplot(121)
    plt.imshow(images, cmap = "gray")
    plt.title("Original Images in Gray")
    plt.subplot(122)
    plt.imshow(img_sift)
    plt.title(f"SIFT Keypoints: {len(keypoints)}")
    plt.axis("off")
    plt.show()
    return keypoints, descriptors

# -- (B) ORB Feature Extraction --
def plot_orb(images):
    """Function detects keypoints and descriptors using ORB"""
    gray = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures = 500) # initialize ORB detector
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    img_orb = cv2.drawKeypoints(images, keypoints, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(figsize = (6, 6))
    plt.subplot(121)
    plt.imshow(images, cmap = "gray")
    plt.title("Original Image")
    plt.subplot(122)
    plt.imshow(img_orb)
    plt.title(f"Orb keypoints: {len(keypoints)}")
    plt.axis("off")
    plt.show()
    return keypoints, descriptors

def main():
    images = load_image_array("bottle", num_images = 1)
    if images is not None:
        for img in images:
            plot_histogram(img)
            plot_edge_detection(img)
            hog_features = plot_Gradient_Hist(img)
            print(f"Features of Hog: {hog_features}")
            print("\n--- SIFT ---")
            sift_keypoints, sift_descriptors = plot_sift(img)
            print(f"SIFT Descriptor Shape: {sift_descriptors.shape}")
            print("\n--- ORB ---")
            orb_keypoints, orb_descriptors = plot_orb(img)
            print(f"ORB Descriptor Shape: {orb_descriptors.shape}")

        
        return

    print("Images Failed to load.\n")


if __name__ == "__main__":
    main()