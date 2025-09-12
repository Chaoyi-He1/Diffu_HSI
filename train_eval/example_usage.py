"""
Simple example demonstrating how to use the train_one_epoch function 
from the train.py module with 1D U-Net diffusion training.
"""

import os
import sys
import logging

# Add parent directory to path to import model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import our training utilities
from train import (
    train_one_epoch, 
    validate_one_epoch, 
    generate_samples,
    create_synthetic_1d_dataset
)

# Import model components
from model.u2net_1d import U2Net1D
from model.diffusion_trainer import DiffusionTrainer


def setup_simple_logger():
    """Setup a simple logger for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def main():
    """
    Example of using train_one_epoch function with all required inputs.
    """
    logger = setup_simple_logger()
    logger.info("Starting 1D U-Net diffusion training example")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Model parameters
    input_channels = 3
    condition_dim = 64
    base_channels = 32  # Smaller for faster training
    sequence_length = 128
    
    # Training parameters
    batch_size = 4
    num_epochs = 3
    learning_rate = 1e-4
    num_samples = 60  # Small dataset for quick demo
    
    # Create model
    logger.info("Creating U2Net1D model...")
    model = U2Net1D(
        input_channels=input_channels,
        condition_dim=condition_dim,
        base_channels=base_channels
    ).to(device)
    
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create diffusion trainer
    logger.info("Creating diffusion trainer...")
    diffusion_trainer = DiffusionTrainer(
        n_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        prediction_type="eps",  # Predict noise
        device=device
    )
    
    # Create dataset
    logger.info("Creating synthetic dataset...")
    data, conditions = create_synthetic_1d_dataset(
        num_samples=num_samples,
        input_channels=input_channels,
        condition_dim=condition_dim,
        sequence_length=sequence_length
    )
    
    # Split into train/validation
    split_idx = int(0.8 * len(data))
    train_data, val_data = data[:split_idx], data[split_idx:]
    train_conditions, val_conditions = conditions[:split_idx], conditions[split_idx:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_data, train_conditions)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_data, val_conditions)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Condition shape: {conditions.shape}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Training loop using train_one_epoch function
    logger.info("Starting training loop...")
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        logger.info(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        
        # Train for one epoch using our function
        train_metrics = train_one_epoch(
            model=model,
            diffusion_trainer=diffusion_trainer,
            dataloader=train_dataloader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            grad_clip=1.0,
            log_interval=5
        )
        
        train_losses.append(train_metrics['loss'])
        
        logger.info(
            f"Training completed - "
            f"Loss: {train_metrics['loss']:.6f}, "
            f"Grad norm: {train_metrics['grad_norm']:.4f}, "
            f"Samples/sec: {train_metrics['samples_per_sec']:.1f}"
        )
        
        # Validate every epoch
        val_metrics = validate_one_epoch(
            model=model,
            diffusion_trainer=diffusion_trainer,
            dataloader=val_dataloader,
            device=device
        )
        
        val_losses.append(val_metrics['val_loss'])
        
        logger.info(f"Validation Loss: {val_metrics['val_loss']:.6f}")
    
    # Print training summary
    logger.info("\n=== Training Summary ===")
    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        logger.info(f"Epoch {i+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Test sample generation
    logger.info("\n=== Testing Sample Generation ===")
    
    # Use first few conditions from validation set
    test_conditions = val_conditions[:2].to(device)
    
    generated_samples = generate_samples(
        model=model,
        diffusion_trainer=diffusion_trainer,
        conditions=test_conditions,
        shape=(2, input_channels, sequence_length),
        device=device,
        num_steps=20,  # Fewer steps for faster generation
        method="ddpm"
    )
    
    logger.info(f"Generated samples shape: {generated_samples.shape}")
    logger.info(f"Generated samples range: [{generated_samples.min():.3f}, {generated_samples.max():.3f}]")
    
    logger.info("\n=== Example completed successfully! ===")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'generated_samples': generated_samples
    }


if __name__ == "__main__":
    results = main()
    print("\nYou can now access:")
    print("- results['model']: Trained model")
    print("- results['train_losses']: Training loss history")
    print("- results['val_losses']: Validation loss history") 
    print("- results['generated_samples']: Generated sample outputs")
