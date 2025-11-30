"""
Inference script for enhancing dark images using trained UNet model.
"""
import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from unet_model import UNet

# Configuration
DATASET_ROOT = "RoadAnomaly_jpg"
CHECKPOINT_DIR = os.path.join(DATASET_ROOT, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "enhance_unet.pth")
IMAGE_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image_path, image_size=IMAGE_SIZE):
    """
    Preprocess image for inference (same as training).
    
    Args:
        image_path: path to input image
        image_size: target size for resizing
    
    Returns:
        tensor: preprocessed image tensor (1, C, H, W) in [0, 1]
        original_image: PIL Image for visualization
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    
    # Transform: resize to square, normalize to [0, 1]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # Converts to [0, 1] and C×H×W
    ])
    
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor, original_image

def postprocess_image(tensor):
    """
    Postprocess model output to uint8 RGB image.
    
    Args:
        tensor: model output tensor (1, C, H, W) in [0, 1]
    
    Returns:
        PIL Image (RGB, uint8)
    """
    # Remove batch dimension and convert to numpy
    tensor = tensor.squeeze(0)  # (C, H, W)
    tensor = tensor.clamp(0, 1)  # Ensure [0, 1]
    
    # Convert to numpy and scale to [0, 255]
    img_array = (tensor.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    
    return Image.fromarray(img_array)

def enhance_image(model, image_path, device=DEVICE):
    """
    Enhance a dark image.
    
    Args:
        model: trained UNet model
        image_path: path to dark input image
        device: device to run inference on
    
    Returns:
        enhanced_image: PIL Image (RGB, uint8)
    """
    # Preprocess
    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Postprocess
    enhanced_image = postprocess_image(output_tensor)
    
    return enhanced_image, original_image

def visualize_comparison(dark_image, enhanced_image, bright_image=None, save_path=None):
    """
    Create a side-by-side visualization.
    
    Args:
        dark_image: PIL Image (dark input)
        enhanced_image: PIL Image (model output)
        bright_image: PIL Image (ground truth, optional)
        save_path: path to save visualization
    """
    # Resize all images to same size for comparison
    target_size = (512, 512)
    dark_image = dark_image.resize(target_size)
    enhanced_image = enhanced_image.resize(target_size)
    
    # Create comparison image
    if bright_image is not None:
        bright_image = bright_image.resize(target_size)
        # Horizontal: dark | enhanced | bright
        total_width = target_size[0] * 3
        comparison = Image.new('RGB', (total_width, target_size[1]))
        comparison.paste(dark_image, (0, 0))
        comparison.paste(enhanced_image, (target_size[0], 0))
        comparison.paste(bright_image, (target_size[0] * 2, 0))
    else:
        # Horizontal: dark | enhanced
        total_width = target_size[0] * 2
        comparison = Image.new('RGB', (total_width, target_size[1]))
        comparison.paste(dark_image, (0, 0))
        comparison.paste(enhanced_image, (target_size[0], 0))
    
    if save_path:
        comparison.save(save_path)
        print(f"Visualization saved to {save_path}")
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description='Enhance dark images using trained UNet')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input dark image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save enhanced image (default: input_enhanced.jpg)')
    parser.add_argument('--checkpoint', type=str, default=BEST_MODEL_PATH,
                       help=f'Path to model checkpoint (default: {BEST_MODEL_PATH})')
    parser.add_argument('--bright', type=str, default=None,
                       help='Path to ground truth bright image (optional, for comparison)')
    parser.add_argument('--visualize', type=str, default=None,
                       help='Path to save comparison visualization')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first using train.py")
        return
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = UNet(n_channels=3, n_classes=3, bilinear=True).to(DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    # Determine output path
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_enhanced.jpg"
    
    # Enhance image
    print(f"Enhancing image: {args.input}")
    enhanced_image, original_image = enhance_image(model, args.input, DEVICE)
    
    # Save enhanced image
    enhanced_image.save(args.output)
    print(f"Enhanced image saved to {args.output}")
    
    # Create visualization if requested
    if args.visualize or args.bright:
        bright_image = None
        if args.bright and os.path.exists(args.bright):
            bright_image = Image.open(args.bright).convert('RGB')
        
        vis_path = args.visualize if args.visualize else args.output.replace('.jpg', '_comparison.jpg')
        visualize_comparison(original_image, enhanced_image, bright_image, vis_path)

if __name__ == "__main__":
    main()

