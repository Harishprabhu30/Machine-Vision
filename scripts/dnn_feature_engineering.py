import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

processed_image_array_dir = Path("/Users/harishprabhu/Desktop/Machine_Vision/Machine-Vision/data/processed")

# Load a pre-trained model ResNet Model (without the final classification layer)
model = models.resnet18(pretrained = True)
model = torch.nn.Sequential(*list(model.children())[: -1]) # Excluding the last layer
model.eval() # set to evaluation mode

# Define image processing (Match ImageNet training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), # resnet input size
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image_array(category, num_images = 5):
    """Loads processed images from NumPy file."""

    file_path = processed_image_array_dir / f"{category}_train.npy"

    if not file_path.exists():
        print(f"File {file_path} not found! Run data ingestion first.")
        return None

    print("File Found! Extracting CNN Features using ResNet-18...")

    images = np.load(file_path)
    return images[:num_images]

def extract_cnn_features(images):
    """Function extracts cnn embedding using resnet18"""
    img_tensor = transform(images).unsqueeze(0) # add batch dimension
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy() # Convert tensor to numpy array

def main():
    
    images = load_image_array("bottle", num_images = 5)
    if images is not None:
        for i, img in enumerate(images):
            features = extract_cnn_features(img)
            print(f"Image {i+1} - CNN Feature Vector Shape: {features.shape}")
        return
    
    print("Images Failed to load.\n")

if __name__ == "__main__":
    main()