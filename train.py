"""
Training script for UNet image enhancement model.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

from unet_model import UNet
from dataset import DarkBrightDataset

# Configuration
DATASET_ROOT = "RoadAnomaly_jpg"
CHECKPOINT_DIR = os.path.join(DATASET_ROOT, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "enhance_unet.pth")

# Training hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

def calculate_psnr(pred, target):
    """Calculate PSNR between predicted and target images."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for dark_batch, bright_batch in pbar:
        dark_batch = dark_batch.to(device)
        bright_batch = bright_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_bright = model(dark_batch)
        
        # Compute loss
        loss = criterion(pred_bright, bright_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        with torch.no_grad():
            psnr = calculate_psnr(pred_bright, bright_batch)
            total_psnr += psnr
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{psnr:.2f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    return avg_loss, avg_psnr

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for dark_batch, bright_batch in pbar:
            dark_batch = dark_batch.to(device)
            bright_batch = bright_batch.to(device)
            
            # Forward pass
            pred_bright = model(dark_batch)
            
            # Compute loss
            loss = criterion(pred_bright, bright_batch)
            
            # Metrics
            total_loss += loss.item()
            psnr = calculate_psnr(pred_bright, bright_batch)
            total_psnr += psnr
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{psnr:.2f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    return avg_loss, avg_psnr

def main():
    print(f"Using device: {DEVICE}")
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = DarkBrightDataset(split='train')
    val_dataset = DarkBrightDataset(split='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("Creating UNet model...")
    model = UNet(n_channels=3, n_classes=3, bilinear=True).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training history
    history = {
        'train_loss': [],
        'train_psnr': [],
        'val_loss': [],
        'val_psnr': []
    }
    
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("-" * 60)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_psnr = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_psnr = validate(model, val_loader, criterion, DEVICE)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_psnr'].append(train_psnr)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f} dB")
        print(f"Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f} dB")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'history': history
            }, BEST_MODEL_PATH)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        print("-" * 60)
    
    # Save final history
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print(f"Best model saved to {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()

