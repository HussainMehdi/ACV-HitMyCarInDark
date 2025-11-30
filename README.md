# Image Enhancement Pipeline: Dark → Bright

This pipeline trains a UNet model to enhance dark road images, converting them to bright versions. The system uses synthetic dark images generated from bright ground truth images.

## Overview

The pipeline consists of the following steps:

1. **Dataset Preparation**: Collect JPG images and create train/val split
2. **Dark Image Generation**: Generate synthetic dark images with deterministic seeds
3. **Model Training**: Train UNet to learn dark → bright mapping
4. **Inference**: Enhance new dark images using the trained model

## Directory Structure

```
.
├── RoadAnomaly_jpg/
│   ├── frames/              # Original bright images (.jpg)
│   ├── frames_dark/          # Generated dark images (created by step 2)
│   ├── checkpoints/          # Model checkpoints (created during training)
│   └── split_indices.json    # Train/val split (created by step 1)
├── create_split.py           # Step 1: Create train/val split
├── generate_dark_images.py   # Step 2: Generate dark images
├── dataset.py                # PyTorch dataset class
├── unet_model.py             # UNet architecture
├── train.py                  # Step 3: Training script
├── inference.py              # Step 4: Inference script
└── requirements.txt          # Python dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Create Train/Val Split

Collect all JPG images and create a reproducible 80/20 train/val split:

```bash
python create_split.py
```

This creates `RoadAnomaly_jpg/split_indices.json` with the split information.

### Step 2: Generate Dark Images

Generate synthetic dark images from bright images using deterministic darkening:

```bash
python generate_dark_images.py
```

This creates `RoadAnomaly_jpg/frames_dark/` with dark versions of all bright images.

**Parameters** (can be modified in the script):
- `GAMMA_MIN = 2.0`, `GAMMA_MAX = 4.0`: Gamma range for darkening
- `NOISE_STD = 0.03`: Standard deviation of additive Gaussian noise
- `GLOBAL_SEED = 42`: Global seed for reproducibility

### Step 3: Train the Model

Train the UNet model to learn dark → bright mapping:

```bash
python train.py
```

**Training Parameters** (can be modified in the script):
- `BATCH_SIZE = 8`: Batch size
- `LEARNING_RATE = 1e-4`: Learning rate for AdamW optimizer
- `NUM_EPOCHS = 50`: Number of training epochs
- `IMAGE_SIZE = 512`: Input/output image size (H=W)

The best model (based on validation loss) is saved to:
- `RoadAnomaly_jpg/checkpoints/enhance_unet.pth`

Training history is saved to:
- `RoadAnomaly_jpg/checkpoints/training_history.json`

### Step 4: Inference

Enhance a dark image using the trained model:

```bash
python inference.py --input <path_to_dark_image> --output <output_path>
```

**Options**:
- `--input`: Path to input dark image (required)
- `--output`: Path to save enhanced image (default: `input_enhanced.jpg`)
- `--checkpoint`: Path to model checkpoint (default: `RoadAnomaly_jpg/checkpoints/enhance_unet.pth`)
- `--bright`: Path to ground truth bright image (optional, for comparison)
- `--visualize`: Path to save side-by-side comparison visualization

**Example**:
```bash
# Basic inference
python inference.py --input RoadAnomaly_jpg/frames_dark/animals01_Guiguinto_railway_station_Calves.jpg

# With comparison visualization
python inference.py \
    --input RoadAnomaly_jpg/frames_dark/animals01_Guiguinto_railway_station_Calves.jpg \
    --bright RoadAnomaly_jpg/frames/animals01_Guiguinto_railway_station_Calves.jpg \
    --visualize comparison.jpg
```

## Model Architecture

The UNet model:
- **Input**: 3-channel RGB dark image (512×512)
- **Output**: 3-channel RGB bright image (512×512)
- **Architecture**: Encoder-decoder with skip connections
- **Loss**: L1 loss between predicted and target bright images
- **Optimizer**: AdamW with learning rate 1e-4

## Reproducibility

All random operations use fixed seeds:
- Global seed: `42` (used for train/val split and dark image generation)
- Per-image seeds: Derived deterministically from filename and global seed
- This ensures the same dark images are generated every time

## Notes

- The `.labels` subdirectories are ignored during image collection
- Dark images are generated deterministically, so re-running `generate_dark_images.py` will produce identical results
- The model checkpoint includes model weights, optimizer state, and training history

