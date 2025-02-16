import os
import shutil
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

file_path = Path("/Users/harishprabhu/Desktop/Machine_Vision/Machine-Vision/data/mvtec_anomaly_detection.tar.xz")
dataset_dir = Path("/Users/harishprabhu/Desktop/Machine_Vision/Machine-Vision/data/raw")
processed_images_output_dir = Path("/Users/harishprabhu/Desktop/Machine_Vision/Machine-Vision/data/processed")
processed_images_output_dir.mkdir(parent = True, exists_ok = True)

# image size
image_size = (256, 256)


def extract_dataset():
    """
    function extracts data into destination folder
    """
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print("Extraction Starting.\n")
        shutil.unpack_archive(file_path, dataset_dir)
        print("Extraction Complete.\n")
    else:
        print("Dataset already exists.\n")

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

def process_dataset():
    """Function"""
    categories = os.listdir(dataset_dir)

    for category in categories:
        if category == "bottle":
            category_path = dataset_dir / category / "train/good"

            if category_path.exists():
                print(f"Processing {category}...")
                images = image_2_array(category_path)
                print("Process Successfull for {category}.")

                # saving numpy array as .npy file for faster access
                np.save(processed_images_output_dir / f"{category}_train.npy", images)
                print(f"Saved: {category}_train.npy at {processed_images_output_dir}")
        else:
            print(f"bottle dataset not found.") 


def main():
    extract_dataset()
    process_dataset()    

if __name__ == "__main__":
    main()