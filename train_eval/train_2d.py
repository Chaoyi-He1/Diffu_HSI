"""
Training utilities for 2D U-Net Diffusion Model without VAE.

This module provides training functions for direct 2D hyperspectral diffusion
without VAE compression. The diffusion model operates directly on the original 
hyperspectral data space.

Key functions:
- train_one_epoch: Train diffusion model on original hyperspectral data
- validate_one_epoch: Validate diffusion model
- generate_samples: Generate samples directly in data space
- train_full_pipeline: Complete training pipeline without VAE
"""

import os
import sys
import time
import logging
from typing import Dict, Optional, Tuple, List, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Import MetricLogger from util.py
from misc.util import MetricLogger, SmoothedValue
from model.diffusion_trainer import DiffusionTrainer


def train_one_epoch(
    model: nn.Module,
    diffusion_trainer: DiffusionTrainer,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
    grad_clip: Optional[float] = None,
    log_interval: int = 200,
    scaler: Optional[torch.amp.autocast] = None
) -> Dict[str, float]:
    """
    Train the model for one epoch using MetricLogger for comprehensive logging.
    
    Args:
        model: The 2D U-Net hyperspectral model to train
        diffusion_trainer: Diffusion trainer with loss computation method
        dataloader: Training data loader yielding (data, conditions) tuples
        optimizer: PyTorch optimizer
        epoch: Current epoch number (0-indexed)
        device: Device to run training on (cuda/cpu)
        grad_clip: Gradient clipping threshold (None to disable)
        log_interval: Log metrics every N steps
        scaler: Optional scaler for mixed-precision training
        
    Returns:
        Dictionary containing training metrics:
        - 'loss': Average loss for the epoch
        - 'learning_rate': Current learning rate
        - 'samples_per_sec': Average samples processed per second
        - 'grad_norm': Final gradient norm
        - 'total_samples': Total number of samples processed
    """
    model.train()
    
    # Create MetricLogger internally
    metric_logger = MetricLogger(delimiter="  ")
    
    # Add learning rate meter
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # Setup header for logging
    header = f'Epoch: [{epoch+1}]'
    
    total_samples = 0
    step_times = []
    
    # Use MetricLogger's log_every method for automatic progress logging
    for step, batch in enumerate(metric_logger.log_every(dataloader, log_interval, header)):
        step_start_time = time.time()
        
        # Unpack batch (expecting data, conditions)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            data, conditions = batch
        else:
            raise ValueError(f"Expected batch to be (data, conditions) tuple, got {type(batch)}")
        
        # Move data to device
        data = data.to(device)  # [B, C, H, W] - original hyperspectral data
        conditions = conditions.to(device)  # [B, sensor_channels, H, W] - sensor response
        batch_size = data.shape[0]
        
        # Compute loss using diffusion trainer directly on original data
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            loss, info = diffusion_trainer.get_loss(model, data, conditions)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping and norm computation
        grad_norm = 0.0
        if grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            # Compute grad norm for monitoring
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** (1. / 2)
        
        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Update metrics
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        total_samples += batch_size
        
        # Compute samples per second
        samples_per_sec = batch_size / step_time if step_time > 0 else 0
        
        # Update MetricLogger with current step metrics
        metric_logger.update(
            loss=loss.item(),
            grad_norm=grad_norm.item(),
            samples_per_sec=samples_per_sec,
            lr=optimizer.param_groups[0]['lr']
        )
    
    # Synchronize metrics across processes if using distributed training
    metric_logger.synchronize_between_processes()
    
    # Compute final averaged metrics
    final_metrics = {
        'loss': metric_logger.loss.global_avg,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'samples_per_sec': metric_logger.samples_per_sec.global_avg,
        'grad_norm': metric_logger.grad_norm.global_avg,
        'total_samples': total_samples
    }
    
    return final_metrics


def validate_one_epoch(
    model: nn.Module,
    diffusion_trainer: Any,
    dataloader: DataLoader,
    device: torch.device,
    scaler: Optional[torch.amp.autocast] = None
) -> Dict[str, float]:
    """
    Validate the model for one epoch using MetricLogger.
    
    Args:
        model: The 2D U-Net hyperspectral model to validate
        diffusion_trainer: Diffusion trainer with loss computation
        dataloader: Validation data loader
        device: Device to run validation on
        scaler: Optional scaler for mixed-precision validation
        
    Returns:
        Dictionary containing validation metrics:
        - 'val_loss': Average validation loss
        - 'val_samples': Total validation samples processed
    """
    model.eval()
    
    # Create MetricLogger internally
    metric_logger = MetricLogger(delimiter="  ")
    
    header = 'Validation:'
    total_samples = 0
    
    with torch.no_grad():
        for batch in metric_logger.log_every(dataloader, 50, header):
            # Unpack batch
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, conditions = batch
            else:
                raise ValueError(f"Expected batch to be (data, conditions) tuple, got {type(batch)}")
            
            # Move data to device
            data = data.to(device)  # [B, C, H, W] - original hyperspectral data
            conditions = conditions.to(device)  # [B, sensor_channels, H, W] - sensor response
            batch_size = data.shape[0]
            
            # Compute loss with optional mixed-precision directly on original data
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                loss, info = diffusion_trainer.get_loss(model, data, conditions)
            
            total_samples += batch_size
            
            # Update MetricLogger
            metric_logger.update(val_loss=loss.item())
    
    # Synchronize metrics across processes if using distributed training
    metric_logger.synchronize_between_processes()
    
    return {
        'val_loss': metric_logger.val_loss.global_avg,
        'val_samples': total_samples
    }


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    diffusion_trainer: Any,
    conditions: torch.Tensor,
    data_shape: Tuple[int, ...],
    device: torch.device,
    num_steps: Optional[int] = None,
    method: str = "ddpm",
    scaler: Optional[torch.amp.autocast] = None
) -> torch.Tensor:
    """
    Generate samples using the trained model directly in data space.
    
    Args:
        model: Trained 2D hyperspectral diffusion model
        diffusion_trainer: Diffusion trainer with sampling capability
        conditions: Condition tensors [B, sensor_channels, H, W]
        data_shape: Shape of data samples to generate [B, spectral_channels, H, W]
        device: Device to run on
        num_steps: Number of sampling steps (None uses trainer default)
        method: Sampling method ('ddpm' or 'ddim')
        scaler: Optional scaler for mixed-precision generation
        
    Returns:
        Generated samples [B, spectral_channels, H, W] in original data space
    """
    model.eval()
    
    print(f'Generating {data_shape[0]} samples with shape {data_shape}')
    
    # Generate samples using the diffusion trainer's sampling method
    with torch.amp.autocast('cuda', enabled=scaler is not None):
        samples = diffusion_trainer.sample(
            model=model,
            cond=conditions,
            shape=data_shape,
            n_steps=num_steps,
            method=method,
            progress=True
        )
    
    print(f'Generation completed. Sample range: [{samples.min():.3f}, {samples.max():.3f}]')
    
    return samples


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    diffusion_config: Optional[dict] = None
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current loss value
        filepath: Path to save checkpoint
        diffusion_config: Optional diffusion trainer configuration
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'diffusion_config': diffusion_config
    }
    torch.save(checkpoint, filepath)
    print(f'Checkpoint saved to {filepath}')


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None
) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f'Checkpoint loaded from {filepath}')
    
    return checkpoint


def train_full_pipeline(
    model: nn.Module,
    diffusion_trainer: DiffusionTrainer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    num_epochs: int,
    device: torch.device,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    save_dir: Optional[str] = None,
    grad_clip: Optional[float] = 1.0,
    val_every: int = 1000,
    save_every: int = 1000,
    generate_every: int = 2000,
    scaler: Optional[torch.amp.autocast] = None,
    log_interval: int = 200
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """
    Complete training pipeline using train_one_epoch function with MetricLogger.
    
    Args:
        model: 2D hyperspectral diffusion model to train
        diffusion_trainer: Diffusion trainer
        train_dataloader: Training data loader
        val_dataloader: Optional validation data loader
        num_epochs: Number of epochs to train
        device: Device to train on
        optimizer: Optional optimizer (creates AdamW if None)
        scheduler: Optional learning rate scheduler
        save_dir: Optional directory to save checkpoints
        grad_clip: Gradient clipping threshold
        val_every: Validate every N epochs
        save_every: Save checkpoint every N epochs
        generate_every: Generate samples every N epochs
        scaler: Optional scaler for mixed-precision training
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    import os
    
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Training history
    train_history = []
    val_history = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\n=== Epoch {epoch+1}/{num_epochs} ===')
        
        # Training
        train_metrics = train_one_epoch(
            model=model,
            diffusion_trainer=diffusion_trainer,
            dataloader=train_dataloader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            grad_clip=grad_clip,
            scaler=scaler,
            log_interval=log_interval
        )
        train_history.append(train_metrics)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
        
        # Validation
        if val_dataloader is not None and epoch % val_every == 0:
            val_metrics = validate_one_epoch(
                model=model,
                diffusion_trainer=diffusion_trainer,
                dataloader=val_dataloader,
                device=device,
                scaler=scaler
            )
            val_history.append(val_metrics)
            
            # Save best model
            if save_dir and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                save_checkpoint(
                    model, optimizer, epoch, val_metrics['val_loss'],
                    os.path.join(save_dir, 'best_model.pth'),
                    diffusion_config=getattr(diffusion_trainer, '__dict__', None)
                )
        
        # Save checkpoint
        if save_dir and epoch % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics['loss'],
                os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                diffusion_config=getattr(diffusion_trainer, '__dict__', None)
            )
        
        # Generate samples
        if save_dir and epoch % generate_every == 0:
            try:
                # Use a few samples from training data as conditions
                sample_batch = next(iter(train_dataloader))
                sample_conditions = sample_batch[1][:4].to(device)  # First 4 conditions
                sample_data = sample_batch[0][:4].to(device)  # First 4 data samples for shape reference
                
                # Generate samples directly in data space
                generated = generate_samples(
                    model=model,
                    diffusion_trainer=diffusion_trainer,
                    conditions=sample_conditions,
                    data_shape=sample_data.shape,
                    device=device,
                    num_steps=50,
                    method="ddpm",
                    scaler=scaler
                )
                
                # Save generated samples
                torch.save(generated.cpu(), 
                          os.path.join(save_dir, f'generated_epoch_{epoch+1}.pt'))
                
            except Exception as e:
                print(f'Failed to generate samples: {e}')
        
        print(
            f'Epoch {epoch+1} completed - '
            f'Train Loss: {train_metrics["loss"]:.6f}, '
            f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
        )
    
    # Save final model
    if save_dir:
        save_checkpoint(
            model, optimizer, num_epochs-1, train_history[-1]['loss'],
            os.path.join(save_dir, 'final_model.pth'),
            diffusion_config=getattr(diffusion_trainer, '__dict__', None)
        )

    return model, train_history


# Example usage functions
def create_synthetic_2d_dataset(
    num_samples: int = 1000,
    spectral_channels: int = 128,
    sensor_channels: int = 16,
    height: int = 64,
    width: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic 2D hyperspectral dataset for demonstration and testing.
    
    Args:
        num_samples: Number of samples to generate
        spectral_channels: Number of hyperspectral channels
        sensor_channels: Number of sensor response channels
        height: Image height
        width: Image width
    
    Returns:
        Tuple of (data, conditions) where:
        - data: [num_samples, spectral_channels, height, width]
        - conditions: [num_samples, sensor_channels, height, width]
    """
    import torch.nn.functional as F
    
    print(f"Creating synthetic 2D hyperspectral dataset with {num_samples} samples...")
    
    data = []
    conditions = []
    
    for i in range(num_samples):
        # Create base 2D pattern with some structure
        x = torch.linspace(-2, 2, width)
        y = torch.linspace(-2, 2, height)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Generate hyperspectral data with spectral correlation
        hyperspectral_image = torch.zeros(spectral_channels, height, width)
        
        # Create spectral signatures (smooth curves across channels)
        wavelengths = torch.linspace(400, 1000, spectral_channels)  # nm
        for ch in range(spectral_channels):
            # Base spatial pattern
            freq_x = torch.rand(1) * 2 + 0.5
            freq_y = torch.rand(1) * 2 + 0.5
            phase_x = torch.rand(1) * 2 * np.pi
            phase_y = torch.rand(1) * 2 * np.pi
            
            # Spectral modulation - smooth variation across wavelengths
            spectral_response = 0.5 + 0.3 * torch.cos(2 * np.pi * wavelengths[ch] / 100) + 0.2 * torch.sin(2 * np.pi * wavelengths[ch] / 150)
            
            # Create pattern with sine waves and radial components
            pattern = (torch.sin(freq_x * X + phase_x) * torch.sin(freq_y * Y + phase_y) +
                      0.3 * torch.exp(-(X**2 + Y**2)/2) * torch.sin(3*torch.atan2(Y, X)) +
                      0.1 * torch.randn(height, width))  # Add noise
            
            hyperspectral_image[ch] = pattern * spectral_response
        
        # Apply some smoothing
        hyperspectral_image = F.conv2d(hyperspectral_image.unsqueeze(0), 
                                     torch.ones(1, 1, 3, 3)/9, 
                                     padding=1, groups=1).squeeze(0)
        
        # Create corresponding sensor response (simulated RGB-like channels)
        sensor_response = torch.zeros(sensor_channels, height, width)
        for s_ch in range(sensor_channels):
            # Each sensor channel responds to a range of spectral channels
            start_idx = (s_ch * spectral_channels) // sensor_channels
            end_idx = ((s_ch + 1) * spectral_channels) // sensor_channels
            
            # Weighted average of corresponding spectral channels
            weights = torch.linspace(0.1, 1.0, end_idx - start_idx)
            for i, spec_idx in enumerate(range(start_idx, end_idx)):
                if spec_idx < spectral_channels:
                    sensor_response[s_ch] += weights[i] * hyperspectral_image[spec_idx]
            
            # Normalize and add noise
            sensor_response[s_ch] = sensor_response[s_ch] / (end_idx - start_idx) + 0.05 * torch.randn(height, width)
        
        data.append(hyperspectral_image)
        conditions.append(sensor_response)
    
    return torch.stack(data), torch.stack(conditions)


if __name__ == "__main__":
    # Example usage
    print("This is a utility module for 2D hyperspectral diffusion training without VAE.")
    print("Available functions:")
    print("- train_one_epoch: Train model for one epoch on original hyperspectral data")
    print("- validate_one_epoch: Validate model for one epoch") 
    print("- generate_samples: Generate samples directly in hyperspectral data space")
    print("- train_full_pipeline: Complete training pipeline without VAE")
    print("- save_checkpoint/load_checkpoint: Checkpoint utilities")
    print("- create_synthetic_2d_dataset: Create synthetic 2D hyperspectral data for testing")
