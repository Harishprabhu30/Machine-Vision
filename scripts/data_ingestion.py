import os
import sys
import shutil
import yaml
import logging
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Get the root directory and add it to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.logger import logger  # Now it should work
from src import config


zip_file_path = Path(config['paths']['zip_file'])
dataset_dir = Path(config['paths']['dataset_dir'])
processed_images_output_dir = Path(config['paths']['processed_data'])
processed_images_output_dir.mkdir(parents = True, exist_ok = True)

# image size
image_size = config['features']['image_size']

def extract_dataset():
    """
    function extracts data into destination folder
    """
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print("Extraction Starting.\n")
        shutil.unpack_archive(zip_file_path, dataset_dir)
        print("Extraction Complete.\n")
        logger.info(f"Zip File Extraction Successful.")

    else:
        print("Dataset already exists.")
        logger.info(f"Zip File Extracted Already. Good to go!")


def image_2_array(folder):
    """Function reads the images, preprocess them and saves as .npy file."""

    images = []
    for filename in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # reading as RGB 
        if img is not None:
            img = cv2.resize(img, image_size) #resize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Convert BGR to RGB
            images.append(img)

    return np.array(images)

def process_dataset(categories):
    """Function"""

    # categories = os.listdir(dataset_dir)
    # bottle_data = Path("/Users/harishprabhu/Desktop/Machine_Vision/Machine-Vision/data/raw/bottle")

    for category in categories:
        category_path = dataset_dir / category / "train/good"
    # category_path = bottle_data / "train/good"

        if category_path.exists():
            print(f"Processing {category}...")
            logger.info(f"Processing {category}.")
            images = image_2_array(category_path)
            print(f"Process Successful for {category}.")
            logger.info(f"Process Successful for {category}.")

            # saving numpy array as .npy file for faster access
            np.save(processed_images_output_dir / f"bottle_train.npy", images)
            print(f"Saved: bottle_train.npy at {processed_images_output_dir}.\n")
            logger.info(f"Saved: bottle_train.npy at {processed_images_output_dir}")


def main():
    
    # Calling defined functions
    logger.info("Starting Data Ingestion.")
    print("-------- Starting Data Ingestion --------")
    extract_dataset()
    categories = config['dataset']['categories']
    process_dataset(categories)
    logger.info(" Data Ingestion Successful.")

if __name__ == "__main__":
    main()