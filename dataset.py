"""
PyTorch dataset for dark→bright image pairs.
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Configuration
DATASET_ROOT = "RoadAnomaly_jpg"
DARK_DIR = os.path.join(DATASET_ROOT, "frames_dark")
SPLIT_FILE = os.path.join(DATASET_ROOT, "split_indices.json")
IMAGE_SIZE = 512

class DarkBrightDataset(Dataset):
    """Dataset for dark→bright image pairs."""
    
    def __init__(self, split='train', image_size=IMAGE_SIZE):
        """
        Args:
            split: 'train' or 'val'
            image_size: target size for resizing (H=W=image_size)
        """
        self.split = split
        self.image_size = image_size
        
        # Load split data
        with open(SPLIT_FILE, 'r') as f:
            split_data = json.load(f)
        
        # Get indices for this split
        if split == 'train':
            indices = split_data['train_indices']
        else:
            indices = split_data['val_indices']
        
        # Build pairs
        self.pairs = []
        bright_image_paths = split_data['image_paths']
        
        for idx in indices:
            bright_path = bright_image_paths[idx]
            filename = os.path.basename(bright_path)
            dark_path = os.path.join(DARK_DIR, filename)
            
            if os.path.exists(dark_path):
                self.pairs.append((dark_path, bright_path))
            else:
                print(f"Warning: Dark image not found: {dark_path}")
        
        # Transform: resize, pad to square, normalize to [0, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Converts to [0, 1] and C×H×W
        ])
        
        print(f"Loaded {len(self.pairs)} {split} pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        dark_path, bright_path = self.pairs[idx]
        
        # Load images
        dark_image = Image.open(dark_path).convert('RGB')
        bright_image = Image.open(bright_path).convert('RGB')
        
        # Apply transforms
        dark_tensor = self.transform(dark_image)
        bright_tensor = self.transform(bright_image)
        
        return dark_tensor, bright_tensor

