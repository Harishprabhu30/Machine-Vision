import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
from pathlib import Path


processed_image_array_dir = Path("/Users/harishprabhu/Desktop/Machine_Vision/Machine-Vision/data/processed/")

def load_and_visualize(category):
    """Function loads .npy file and plot random 5 images from the specified image_2_array converted data"""

    file_path = processed_image_array_dir / f"{category}_train.npy"
    if not file_path.exists():
        print(f"file {category} not found. Run Data Ingestion first!!\n")
        return

    print("File Found! Visualizing Process Started...\n")

    # Load images
    images = np.load(file_path)

    # Generating Random Indices value for selection of random images
    random_indices = np.random.choice(len(images), size = 5, replace = False)

    # plot 5 random images
    fig, axes = plt.subplots(1, 5, figsize = (15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(images[random_indices[i]])
        ax.axis("off")
    plt.show()

def main():
    load_and_visualize("bottle") # add a for loop for visualizaing images of all folders


if __name__ == "__main__":
    main()