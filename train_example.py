"""
Example training script for U2Net Hyperspectral Diffusion Model

This script demonstrates how to integrate the U2NetHyperspectral model
into a diffusion training pipeline for hyperspectral image generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import math

from model.u2net_hyperspectral import U2NetHyperspectral


class HyperspectralDiffusion:
    """
    Simple diffusion scheduler for hyperspectral image generation.
    Implements basic DDPM (Denoising Diffusion Probabilistic Models) schedule.
    """
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
    
    def add_noise(self, x_0, t, noise=None):
        """
        Add noise to clean images according to diffusion schedule.
        
        Args:
            x_0: Clean images [B, L, H, W]
            t: Timesteps [B]
            noise: Optional noise tensor [B, L, H, W]
        
        Returns:
            Noisy images x_t [B, L, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise, noise
    
    def to(self, device):
        """Move scheduler tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)
        return self


def create_synthetic_dataset(num_samples=1000, spectral_channels=128, sensor_channels=16, 
                           height=64, width=64):
    """
    Create synthetic hyperspectral dataset for demonstration.
    
    In practice, this would load real hyperspectral images and corresponding sensor responses.
    """
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    # Generate synthetic hyperspectral images
    # Simulate spectral signatures with smooth variations
    hyperspectral_images = []
    sensor_responses = []
    
    for i in range(num_samples):
        # Create base spectral signature
        base_spectrum = torch.randn(spectral_channels)
        base_spectrum = F.conv1d(base_spectrum.unsqueeze(0).unsqueeze(0), 
                                torch.ones(1, 1, 5)/5, padding=2).squeeze()
        
        # Create spatial variations
        spatial_pattern = torch.randn(height, width)
        spatial_pattern = F.conv2d(spatial_pattern.unsqueeze(0).unsqueeze(0),
                                 torch.ones(1, 1, 5, 5)/25, padding=2).squeeze()
        
        # Combine spectral and spatial information
        hyperspectral = base_spectrum.unsqueeze(-1).unsqueeze(-1) * (1 + 0.3 * spatial_pattern)
        hyperspectral = hyperspectral + 0.1 * torch.randn_like(hyperspectral)
        
        # Create corresponding sensor response (simplified)
        # In practice, this would be derived from sensor characteristics
        sensor_response = torch.randn(sensor_channels, height, width) * 0.5
        
        hyperspectral_images.append(hyperspectral)
        sensor_responses.append(sensor_response)
    
    return torch.stack(hyperspectral_images), torch.stack(sensor_responses)


def train_hyperspectral_diffusion(model, dataloader, diffusion_scheduler, 
                                 num_epochs=100, device='cuda', lr=1e-4):
    """
    Train the hyperspectral diffusion model.
    """
    model = model.to(device)
    diffusion_scheduler = diffusion_scheduler.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    model.train()
    total_steps = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (hyperspectral_images, sensor_responses) in enumerate(pbar):
            hyperspectral_images = hyperspectral_images.to(device)
            sensor_responses = sensor_responses.to(device)
            batch_size = hyperspectral_images.shape[0]
            
            # Sample random timesteps for each image in the batch
            t = torch.randint(0, diffusion_scheduler.num_timesteps, (batch_size,), 
                            device=device, dtype=torch.long)
            
            # Add noise to the images
            noisy_images, noise = diffusion_scheduler.add_noise(hyperspectral_images, t)
            
            # Predict the noise
            predicted_noise = model(noisy_images, t, sensor_responses)
            
            # Compute loss (simple MSE between predicted and actual noise)
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            total_steps += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.6f}'
            })
        
        print(f'Epoch {epoch+1} completed. Average loss: {epoch_loss/len(dataloader):.6f}')
    
    return model


@torch.no_grad()
def generate_samples(model, sensor_responses, diffusion_scheduler, num_steps=50, device='cuda'):
    """
    Generate hyperspectral images from sensor responses using DDPM sampling.
    
    This is a simplified sampling procedure. In practice, you might want to use
    more sophisticated samplers like DDIM, DPM-Solver, etc.
    """
    model.eval()
    B, S, H, W = sensor_responses.shape
    spectral_channels = model.spectral_channels
    
    # Start with pure noise
    x = torch.randn(B, spectral_channels, H, W, device=device)
    
    # Reverse diffusion process
    timesteps = torch.linspace(diffusion_scheduler.num_timesteps-1, 0, num_steps, dtype=torch.long)
    
    for i, t_val in enumerate(tqdm(timesteps, desc='Generating')):
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x, t, sensor_responses)
        
        # Simple DDPM step (this is a simplified version)
        alpha_t = diffusion_scheduler.alphas[t_val]
        alpha_cumprod_t = diffusion_scheduler.alpha_cumprod[t_val]
        alpha_cumprod_t_minus_1 = diffusion_scheduler.alpha_cumprod[t_val-1] if t_val > 0 else 1.0
        
        # Compute x_{t-1}
        coeff1 = 1.0 / torch.sqrt(alpha_t)
        coeff2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)
        
        x = coeff1 * (x - coeff2 * predicted_noise)
        
        # Add noise if not the final step
        if t_val > 0:
            beta_t = diffusion_scheduler.betas[t_val]
            variance = ((1.0 - alpha_cumprod_t_minus_1) / (1.0 - alpha_cumprod_t)) * beta_t
            noise = torch.randn_like(x) * torch.sqrt(variance)
            x = x + noise
    
    return x


def main():
    """
    Main training and demonstration function.
    """
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    spectral_channels = 128
    sensor_channels = 16
    base_channels = 64
    
    # Training parameters
    batch_size = 4
    num_epochs = 20
    learning_rate = 1e-4
    num_samples = 200  # Small dataset for demo
    
    # Create model and diffusion scheduler
    model = U2NetHyperspectral(spectral_channels, sensor_channels, base_channels)
    diffusion_scheduler = HyperspectralDiffusion(num_timesteps=1000)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Create synthetic dataset
    hyperspectral_data, sensor_data = create_synthetic_dataset(
        num_samples=num_samples,
        spectral_channels=spectral_channels,
        sensor_channels=sensor_channels,
        height=64,
        width=64
    )
    
    # Create dataloader
    dataset = TensorDataset(hyperspectral_data, sensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Hyperspectral shape: {hyperspectral_data.shape}")
    print(f"Sensor response shape: {sensor_data.shape}")
    
    # Train the model
    print("\nStarting training...")
    trained_model = train_hyperspectral_diffusion(
        model, dataloader, diffusion_scheduler,
        num_epochs=num_epochs, device=device, lr=learning_rate
    )
    
    # Generate samples
    print("\nGenerating samples...")
    test_sensor_responses = sensor_data[:4].to(device)  # Use first 4 samples
    generated_images = generate_samples(
        trained_model, test_sensor_responses, diffusion_scheduler,
        num_steps=50, device=device
    )
    
    print(f"Generated images shape: {generated_images.shape}")
    print("Training and generation completed!")
    
    # Save model (optional)
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_config': {
            'spectral_channels': spectral_channels,
            'sensor_channels': sensor_channels,
            'base_channels': base_channels
        }
    }, 'u2net_hyperspectral_checkpoint.pth')
    print("Model checkpoint saved!")


if __name__ == "__main__":
    main()
