import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Get the root directory and add it to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from src.logger import logger  # Now it should work
from src import config

processed_image_array_dir = Path(config['paths']['processed_data'])

def load_and_visualize(category):
    """Function loads .npy file and plot random 5 images from the specified image_2_array converted data"""

    file_path = processed_image_array_dir / f"{category}_train.npy"
    if not file_path.exists():
        print(f"file {category}_train.npy not found. Run Data Ingestion first!!\n")
        logger.info(f"{category}_train.npy not found. Run Data Ingestion first!!")
        return

    print("File Found!")
    logger.info(f"{category}_train.npy found. Loading.")


    # Load images and wraping it with tqdm to display progress bar
    with tqdm(total = 1, desc = "Loading Data", unit = "file") as pbar:
        images = np.load(file_path)
        pbar.update(1) # Mark as completed
    
    print("Loading Complete! Visualization Process Started...\n")
    logger.info(f"Loading Complete! Visualization Process Started.")


    # Generating Random Indices value for selection of random images
    random_indices = np.random.choice(len(images), size = 5, replace = False)

    # plot 5 random images
    fig, axes = plt.subplots(1, 5, figsize = (15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(images[random_indices[i]])
        ax.axis("off")
    plt.show()

def main():
    
    logger.info("Starting Visualizations.")
    print("-------- Starting Visualizations --------")
    categories = config['dataset']['categories']
    for category in categories:
        load_and_visualize(category) # add a for loop for visualizaing images of all folders!
    
    # logger.info(" Visualization Successful.")


if __name__ == "__main__":
    main()