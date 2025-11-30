"""
Script to collect all .jpg images and create a reproducible train/val split.
"""
import os
import json
import random
from pathlib import Path

# Configuration
DATASET_ROOT = "RoadAnomaly_jpg"
FRAMES_DIR = os.path.join(DATASET_ROOT, "frames")
GLOBAL_SEED = 42
SPLIT_FILE = os.path.join(DATASET_ROOT, "split_indices.json")
TRAIN_RATIO = 0.8

def collect_bright_images(frames_dir):
    """Recursively collect all .jpg files, ignoring .labels directories."""
    bright_image_paths = []
    
    for root, dirs, files in os.walk(frames_dir):
        # Skip .labels directories
        dirs[:] = [d for d in dirs if not d.endswith('.labels')]
        
        for file in files:
            if file.lower().endswith('.jpg'):
                full_path = os.path.abspath(os.path.join(root, file))
                bright_image_paths.append(full_path)
    
    return sorted(bright_image_paths)

def create_train_val_split(image_paths, train_ratio=0.8, seed=42):
    """Create reproducible train/val split."""
    random.seed(seed)
    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * train_ratio)
    train_indices = sorted(indices[:split_idx])
    val_indices = sorted(indices[split_idx:])
    
    return train_indices, val_indices

def main():
    print("Collecting bright images...")
    bright_image_paths = collect_bright_images(FRAMES_DIR)
    print(f"Found {len(bright_image_paths)} .jpg images")
    
    print("Creating train/val split...")
    train_indices, val_indices = create_train_val_split(
        bright_image_paths, 
        train_ratio=TRAIN_RATIO, 
        seed=GLOBAL_SEED
    )
    
    print(f"Train: {len(train_indices)} images")
    print(f"Val: {len(val_indices)} images")
    
    # Save split information
    split_data = {
        "global_seed": GLOBAL_SEED,
        "train_ratio": TRAIN_RATIO,
        "total_images": len(bright_image_paths),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "image_paths": bright_image_paths  # Store full paths for convenience
    }
    
    with open(SPLIT_FILE, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"Split saved to {SPLIT_FILE}")

if __name__ == "__main__":
    main()

