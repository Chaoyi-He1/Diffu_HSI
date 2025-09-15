"""
Training and evaluation utilities for Hyperspectral VAE

This module provides independent training and evaluation functions for the HyperspectralVAE.
It includes comprehensive training loops, validation, metrics computation, and model saving/loading.
"""

import os
import sys
import time
import logging
from typing import Dict, Optional, Tuple, List, Any
import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path to import model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import VAE model and loss functions
from model.hyperspectral_vae import HyperspectralVAE, vae_loss
from misc.util import MetricLogger, SmoothedValue


def setup_logger(name: str = "vae_training", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger for VAE training."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def compute_vae_metrics(x_recon: torch.Tensor, x_orig: torch.Tensor, 
                       mean: torch.Tensor, logvar: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive VAE metrics.
    
    Args:
        x_recon: Reconstructed images [B, L, H, W]
        x_orig: Original images [B, L, H, W]
        mean: Latent means [B, C, H/8, W/8]
        logvar: Latent log variances [B, C, H/8, W/8]
        
    Returns:
        Dictionary of computed metrics
    """
    with torch.no_grad():
        batch_size = x_orig.size(0)
        
        # Reconstruction metrics
        mse = F.mse_loss(x_recon, x_orig, reduction='mean')
        mae = F.l1_loss(x_recon, x_orig, reduction='mean')
        
        # PSNR (Peak Signal-to-Noise Ratio)
        # Assuming pixel values are in [0, 1] range
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        
        # Spectral metrics (per-band comparison)
        spectral_mse = torch.mean(torch.mean((x_recon - x_orig) ** 2, dim=(0, 2, 3)))
        spectral_mae = torch.mean(torch.mean(torch.abs(x_recon - x_orig), dim=(0, 2, 3)))
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / mean.numel()
        
        # Latent space metrics
        latent_mean_norm = torch.mean(torch.norm(mean.view(batch_size, -1), dim=1))
        latent_std_mean = torch.mean(torch.exp(0.5 * logvar))
        
        # Reconstruction quality per sample
        per_sample_mse = torch.mean(
            (x_recon - x_orig).view(batch_size, -1) ** 2, dim=1
        )
        reconstruction_variance = torch.var(per_sample_mse)
        
        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'psnr': psnr.item(),
            'spectral_mse': spectral_mse.item(),
            'spectral_mae': spectral_mae.item(),
            'kl_divergence': kl_div.item(),
            'latent_mean_norm': latent_mean_norm.item(),
            'latent_std_mean': latent_std_mean.item(),
            'reconstruction_variance': reconstruction_variance.item()
        }


def train_vae_one_epoch(
    model: HyperspectralVAE,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
    kl_weight: float = 1e-6,
    grad_clip: Optional[float] = None,
    log_interval: int = 10,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Train the VAE for one epoch.
    
    Args:
        model: HyperspectralVAE model
        dataloader: Training data loader
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        device: Device to run training on
        kl_weight: Weight for KL divergence loss
        grad_clip: Gradient clipping threshold
        log_interval: Log metrics every N steps
        logger: Logger instance
        
    Returns:
        Dictionary containing training metrics
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    num_batches = 0
    
    # Handle different dataloader formats
    for batch_idx, batch_data in enumerate(metric_logger.log_every(dataloader, log_interval, header)):
        # Extract data based on format
        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
            data = batch_data[0]  # First element is the data
        else:
            data = batch_data
            
        data = data.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        x_recon, mean, logvar = model(data, sample=True)
        
        # Compute loss
        total_loss, recon_loss, kl_loss = vae_loss(
            x_recon, data, mean, logvar, kl_weight=kl_weight
        )
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Update metrics
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        num_batches += 1
        
        # Update metric logger
        metric_logger.update(
            total_loss=total_loss,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            lr=optimizer.param_groups[0]["lr"]
        )
        
        # Compute additional metrics every log_interval
        if batch_idx % log_interval == 0:
            with torch.no_grad():
                metrics = compute_vae_metrics(x_recon, data, mean, logvar)
                metric_logger.update(**{f"train_{k}": v for k, v in metrics.items()})
    
    # Synchronize metrics across processes if using distributed training
    metric_logger.synchronize_between_processes()
    
    # Calculate epoch averages
    epoch_metrics = {
        'total_loss': total_loss_sum / num_batches,
        'recon_loss': recon_loss_sum / num_batches,
        'kl_loss': kl_loss_sum / num_batches,
        'lr': optimizer.param_groups[0]["lr"]
    }
    
    if logger:
        logger.info(f"Training - {header}")
        logger.info(f"Average losses - Total: {epoch_metrics['total_loss']:.6f}, "
                   f"Recon: {epoch_metrics['recon_loss']:.6f}, "
                   f"KL: {epoch_metrics['kl_loss']:.6f}")
    
    return epoch_metrics


def validate_vae_one_epoch(
    model: HyperspectralVAE,
    dataloader: DataLoader,
    device: torch.device,
    kl_weight: float = 1e-6,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Validate the VAE for one epoch.
    
    Args:
        model: HyperspectralVAE model
        dataloader: Validation data loader
        device: Device to run validation on
        kl_weight: Weight for KL divergence loss
        logger: Logger instance
        
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    all_metrics = []
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in metric_logger.log_every(dataloader, 10, header):
            # Extract data based on format
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
                data = batch_data[0]
            else:
                data = batch_data
                
            data = data.to(device)
            
            # Forward pass
            x_recon, mean, logvar = model(data, sample=False)  # Use mean for validation
            
            # Compute loss
            total_loss, recon_loss, kl_loss = vae_loss(
                x_recon, data, mean, logvar, kl_weight=kl_weight
            )
            
            # Update running sums
            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1
            
            # Compute comprehensive metrics
            metrics = compute_vae_metrics(x_recon, data, mean, logvar)
            all_metrics.append(metrics)
            
            # Update metric logger
            metric_logger.update(
                total_loss=total_loss,
                recon_loss=recon_loss,
                kl_loss=kl_loss,
                **{f"val_{k}": v for k, v in metrics.items()}
            )
    
    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    
    # Calculate epoch averages
    epoch_metrics = {
        'total_loss': total_loss_sum / num_batches,
        'recon_loss': recon_loss_sum / num_batches,
        'kl_loss': kl_loss_sum / num_batches,
    }
    
    # Average comprehensive metrics
    if all_metrics:
        for key in all_metrics[0].keys():
            epoch_metrics[f'val_{key}'] = np.mean([m[key] for m in all_metrics])
    
    if logger:
        logger.info(f"Validation - {header}")
        logger.info(f"Average losses - Total: {epoch_metrics['total_loss']:.6f}, "
                   f"Recon: {epoch_metrics['recon_loss']:.6f}, "
                   f"KL: {epoch_metrics['kl_loss']:.6f}")
        logger.info(f"PSNR: {epoch_metrics.get('val_psnr', 0):.2f}, "
                   f"MAE: {epoch_metrics.get('val_mae', 0):.6f}")
    
    return epoch_metrics


def save_vae_checkpoint(
    model: HyperspectralVAE,
    optimizer: optim.Optimizer,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    save_dir: str,
    is_best: bool = False
):
    """Save VAE model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'model_config': {
            'spectral_channels': model.spectral_channels,
            'latent_channels': model.latent_channels,
            'base_channels': model.base_channels
        }
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, f'vae_checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if applicable
    if is_best:
        best_path = os.path.join(save_dir, 'vae_best_model.pth')
        torch.save(checkpoint, best_path)
    
    # Always save latest
    latest_path = os.path.join(save_dir, 'vae_latest.pth')
    torch.save(checkpoint, latest_path)


def load_vae_checkpoint(
    checkpoint_path: str,
    model: HyperspectralVAE,
    optimizer: Optional[optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    """Load VAE model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    train_metrics = checkpoint.get('train_metrics', {})
    val_metrics = checkpoint.get('val_metrics', {})
    
    return epoch, train_metrics, val_metrics


def generate_vae_samples(
    model: HyperspectralVAE,
    num_samples: int,
    latent_shape: Tuple[int, int, int],  # (C, H, W) for latent space
    device: torch.device,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Generate samples from the VAE by sampling from the latent space.
    
    Args:
        model: Trained HyperspectralVAE model
        num_samples: Number of samples to generate
        latent_shape: Shape of latent space (C, H, W)
        device: Device to run on
        temperature: Temperature for sampling (higher = more diverse)
        
    Returns:
        Generated hyperspectral images [num_samples, L, H*8, W*8]
    """
    model.eval()
    
    with torch.no_grad():
        # Sample from unit Gaussian
        z = torch.randn(num_samples, *latent_shape, device=device) * temperature
        
        # Decode to get samples
        samples = model.decode(z)
        
    return samples


def interpolate_vae_latents(
    model: HyperspectralVAE,
    x1: torch.Tensor,
    x2: torch.Tensor,
    num_steps: int = 10,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Interpolate between two images in the latent space.
    
    Args:
        model: Trained HyperspectralVAE model
        x1, x2: Input images to interpolate between [1, L, H, W]
        num_steps: Number of interpolation steps
        device: Device to run on
        
    Returns:
        Interpolated images [num_steps, L, H, W]
    """
    model.eval()
    
    with torch.no_grad():
        # Encode both images
        mean1, _ = model.encode(x1.to(device))
        mean2, _ = model.encode(x2.to(device))
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, num_steps, device=device)
        
        interpolated_images = []
        for alpha in alphas:
            # Interpolate in latent space
            z_interp = (1 - alpha) * mean1 + alpha * mean2
            
            # Decode interpolated latent
            x_interp = model.decode(z_interp)
            interpolated_images.append(x_interp)
        
        return torch.cat(interpolated_images, dim=0)


def create_synthetic_hyperspectral_dataset(
    num_samples: int,
    spectral_channels: int,
    height: int,
    width: int,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Create synthetic hyperspectral dataset for testing.
    
    Args:
        num_samples: Number of samples to generate
        spectral_channels: Number of spectral bands
        height: Image height
        width: Image width
        device: Device to create data on
        
    Returns:
        Synthetic hyperspectral data [num_samples, spectral_channels, height, width]
    """
    # Create synthetic data with spatial and spectral patterns
    data = torch.randn(num_samples, spectral_channels, height, width, device=device)
    
    # Add some structure to make it more realistic
    for i in range(num_samples):
        # Add spatial gradients
        for c in range(spectral_channels):
            # Spatial structure
            x_grad = torch.linspace(-1, 1, width, device=device)
            y_grad = torch.linspace(-1, 1, height, device=device)
            X, Y = torch.meshgrid(x_grad, y_grad, indexing='xy')
            
            # Spectral variation
            spectral_factor = torch.sin(torch.tensor(c * np.pi / spectral_channels))
            
            # Combine patterns
            pattern = (X * Y * spectral_factor + 
                      0.5 * torch.sin(3 * X) * torch.cos(3 * Y) + 
                      0.3 * torch.randn_like(X))
            
            data[i, c] = pattern
    
    # Normalize to [0, 1]
    data = (data - data.min()) / (data.max() - data.min())
    
    return data


def train_vae_full(
    model: HyperspectralVAE,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    kl_weight: float = 1e-6,
    weight_decay: float = 1e-5,
    grad_clip: Optional[float] = 1.0,
    save_dir: str = "./vae_checkpoints",
    device: torch.device = torch.device('cpu'),
    log_file: Optional[str] = None,
    save_freq: int = 10,
    val_freq: int = 5
) -> Dict[str, List[float]]:
    """
    Complete VAE training loop with validation and checkpointing.
    
    Args:
        model: HyperspectralVAE model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        kl_weight: Weight for KL divergence loss
        weight_decay: Weight decay for optimizer
        grad_clip: Gradient clipping threshold
        save_dir: Directory to save checkpoints
        device: Device to run training on
        log_file: Path to log file
        save_freq: Save checkpoint every N epochs
        val_freq: Run validation every N epochs
        
    Returns:
        Dictionary containing training history
    """
    # Setup logger
    logger = setup_logger("vae_training", log_file)
    logger.info("Starting VAE training")
    logger.info(f"Device: {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'val_psnr': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training
        train_metrics = train_vae_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            kl_weight=kl_weight,
            grad_clip=grad_clip,
            logger=logger
        )
        
        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['train_recon_loss'].append(train_metrics['recon_loss'])
        history['train_kl_loss'].append(train_metrics['kl_loss'])
        history['learning_rate'].append(train_metrics['lr'])
        
        # Validation
        val_metrics = {}
        if val_dataloader is not None and epoch % val_freq == 0:
            val_metrics = validate_vae_one_epoch(
                model=model,
                dataloader=val_dataloader,
                device=device,
                kl_weight=kl_weight,
                logger=logger
            )
            
            history['val_loss'].append(val_metrics['total_loss'])
            history['val_recon_loss'].append(val_metrics['recon_loss'])
            history['val_kl_loss'].append(val_metrics['kl_loss'])
            history['val_psnr'].append(val_metrics.get('val_psnr', 0))
            
            # Learning rate scheduling
            scheduler.step(val_metrics['total_loss'])
            
            # Check if best model
            is_best = val_metrics['total_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['total_loss']
                logger.info(f"New best validation loss: {best_val_loss:.6f}")
        else:
            is_best = False
        
        # Save checkpoint
        if epoch % save_freq == 0 or epoch == num_epochs - 1 or is_best:
            save_vae_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                save_dir=save_dir,
                is_best=is_best
            )
        
        # Log epoch summary
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        logger.info(f"Epoch {epoch}/{num_epochs-1} completed in {epoch_time:.2f}s "
                   f"(Total: {total_time:.2f}s)")
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


if __name__ == "__main__":
    """Example usage of VAE training functions."""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data parameters
    num_samples = 100
    spectral_channels = 128
    height, width = 64, 64
    batch_size = 8
    
    # Model parameters
    latent_channels = 8
    base_channels = 64
    
    # Training parameters
    num_epochs = 50
    learning_rate = 1e-4
    kl_weight = 1e-6
    
    # Create synthetic dataset
    print("Creating synthetic hyperspectral dataset...")
    data = create_synthetic_hyperspectral_dataset(
        num_samples=num_samples,
        spectral_channels=spectral_channels,
        height=height,
        width=width,
        device=device
    )
    
    # Split into train/validation
    split_idx = int(0.8 * num_samples)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print("Creating HyperspectralVAE model...")
    model = HyperspectralVAE(
        spectral_channels=spectral_channels,
        latent_channels=latent_channels,
        base_channels=base_channels
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    print("Starting training...")
    history = train_vae_full(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        kl_weight=kl_weight,
        device=device,
        save_dir="./vae_test_checkpoints",
        log_file="vae_training.log"
    )
    
    print("Training completed!")
    print(f"Final training loss: {history['train_loss'][-1]:.6f}")
    if history['val_loss']:
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
