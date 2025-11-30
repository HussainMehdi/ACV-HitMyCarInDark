"""
Script to generate synthetic dark images from bright images using deterministic darkening.
"""
import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

# Configuration
DATASET_ROOT = "RoadAnomaly_jpg"
FRAMES_DIR = os.path.join(DATASET_ROOT, "frames")
DARK_DIR = os.path.join(DATASET_ROOT, "frames_dark")
SPLIT_FILE = os.path.join(DATASET_ROOT, "split_indices.json")
GLOBAL_SEED = 42

# Darkening parameters
GAMMA_MIN = 2.0
GAMMA_MAX = 4.0
NOISE_STD = 0.03

def get_image_seed(filename, global_seed):
    """Derive a per-image seed from filename and global seed."""
    # Remove extension and hash
    name_without_ext = os.path.splitext(os.path.basename(filename))[0]
    # Simple hash function
    hash_val = hash(name_without_ext) & 0xFFFFFFFF
    image_seed = (hash_val ^ global_seed) & 0xFFFFFFFF
    return int(image_seed)

def darken_image(bright_image, gamma, noise_std, seed):
    """
    Deterministically darken an image using gamma correction and additive noise.
    
    Args:
        bright_image: PIL Image (RGB)
        gamma: gamma value for darkening
        noise_std: standard deviation of Gaussian noise
        seed: random seed for noise generation
    
    Returns:
        PIL Image (RGB, uint8)
    """
    # Convert to float32 [0, 1]
    img_array = np.array(bright_image, dtype=np.float32) / 255.0
    
    # Apply gamma darkening
    dark = np.power(img_array, gamma)
    
    # Generate deterministic noise
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_std, dark.shape).astype(np.float32)
    
    # Add noise and clip
    dark = np.clip(dark + noise, 0.0, 1.0)
    
    # Convert back to uint8 [0, 255]
    dark_uint8 = (dark * 255.0).astype(np.uint8)
    
    return Image.fromarray(dark_uint8)

def sample_gamma(image_seed, gamma_min=GAMMA_MIN, gamma_max=GAMMA_MAX):
    """Sample gamma deterministically based on image seed."""
    rng = np.random.RandomState(image_seed)
    gamma = rng.uniform(gamma_min, gamma_max)
    return gamma

def generate_dark_images():
    """Generate dark images for all bright images."""
    # Load split file to get image paths
    if not os.path.exists(SPLIT_FILE):
        print(f"Error: {SPLIT_FILE} not found. Run create_split.py first.")
        return
    
    with open(SPLIT_FILE, 'r') as f:
        split_data = json.load(f)
    
    bright_image_paths = split_data["image_paths"]
    
    # Create output directory
    os.makedirs(DARK_DIR, exist_ok=True)
    
    print(f"Generating dark images for {len(bright_image_paths)} images...")
    
    for idx, bright_path in enumerate(bright_image_paths):
        # Compute dark path
        filename = os.path.basename(bright_path)
        dark_path = os.path.join(DARK_DIR, filename)
        
        # Skip if already exists (optional: set to False to overwrite)
        if os.path.exists(dark_path):
            print(f"[{idx+1}/{len(bright_image_paths)}] Skipping {filename} (already exists)")
            continue
        
        try:
            # Load bright image
            bright_image = Image.open(bright_path).convert('RGB')
            
            # Compute per-image seed
            image_seed = get_image_seed(bright_path, GLOBAL_SEED)
            
            # Sample gamma deterministically
            gamma = sample_gamma(image_seed, GAMMA_MIN, GAMMA_MAX)
            
            # Darken image
            dark_image = darken_image(bright_image, gamma, NOISE_STD, image_seed)
            
            # Save dark image
            dark_image.save(dark_path, quality=95)
            
            print(f"[{idx+1}/{len(bright_image_paths)}] Generated {filename} (gamma={gamma:.2f})")
            
        except Exception as e:
            print(f"Error processing {bright_path}: {e}")
            continue
    
    print(f"Dark images saved to {DARK_DIR}")

if __name__ == "__main__":
    generate_dark_images()

