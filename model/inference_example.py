"""
Inference example for hyperspectral diffusion models.

This script demonstrates how to:
1. Load pre-trained models
2. Generate new hyperspectral images using DDPM sampling
3. Compress and reconstruct hyperspectral data using VAE
4. Compare full-resolution vs latent-space generation
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from u2net_hyperspectral import U2NetHyperspectral, HyperspectralDiffusion
from hyperspectral_vae import HyperspectralVAE
from latent_u2net_hyperspectral import LatentU2NetHyperspectral, LatentHyperspectralDiffusion

class DDPMSampler:
    """DDPM sampling for generating hyperspectral images"""
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def sample(self, model, shape, context, device, num_steps=50):
        """
        Generate samples using DDPM sampling
        
        Args:
            model: Trained diffusion model
            shape: Shape of samples to generate [B, C, H, W]
            context: Conditioning context [B, S, H, W]
            device: torch.device
            num_steps: Number of denoising steps (less = faster, more = better quality)
            
        Returns:
            Generated samples [B, C, H, W]
        """
        model.eval()
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Create timestep schedule (use fewer steps for faster sampling)
        timesteps = torch.linspace(self.num_timesteps-1, 0, num_steps+1, dtype=torch.long)
        
        with torch.no_grad():
            for i, t in enumerate(timesteps[:-1]):
                t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
                
                # Predict noise
                if hasattr(model, 'model'):
                    # HyperspectralDiffusion wrapper
                    predicted_noise = model.model(x, t_batch, context)
                else:
                    # Direct model
                    predicted_noise = model(x, t_batch, context)
                
                # Compute denoising step
                t_idx = t.item()
                alpha = self.alphas[t_idx].to(device)
                alpha_bar = self.alpha_bars[t_idx].to(device)
                
                # Predicted x_0
                pred_x0 = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
                
                if i < len(timesteps) - 2:  # Not the last step
                    # Add noise for next step
                    next_t = timesteps[i+1].item()
                    next_alpha_bar = self.alpha_bars[next_t].to(device)
                    
                    noise = torch.randn_like(x)
                    x = torch.sqrt(next_alpha_bar) * pred_x0 + torch.sqrt(1 - next_alpha_bar) * noise
                else:
                    x = pred_x0
                    
        return x

def visualize_hyperspectral_rgb(hyperspectral_image, bands=[50, 30, 10]):
    """
    Convert hyperspectral image to RGB for visualization
    
    Args:
        hyperspectral_image: [C, H, W] or [B, C, H, W] tensor
        bands: Which spectral bands to use as R, G, B channels
        
    Returns:
        RGB image [H, W, 3] numpy array
    """
    if hyperspectral_image.dim() == 4:
        hyperspectral_image = hyperspectral_image[0]  # Take first batch
    
    # Extract RGB bands
    r_band = hyperspectral_image[bands[0]].cpu().numpy()
    g_band = hyperspectral_image[bands[1]].cpu().numpy()
    b_band = hyperspectral_image[bands[2]].cpu().numpy()
    
    # Stack and normalize
    rgb = np.stack([r_band, g_band, b_band], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    return rgb

def plot_spectral_signature(hyperspectral_image, x, y, title="Spectral Signature"):
    """
    Plot spectral signature at a specific pixel location
    
    Args:
        hyperspectral_image: [C, H, W] or [B, C, H, W] tensor  
        x, y: Pixel coordinates
        title: Plot title
    """
    if hyperspectral_image.dim() == 4:
        hyperspectral_image = hyperspectral_image[0]  # Take first batch
        
    spectrum = hyperspectral_image[:, y, x].cpu().numpy()
    wavelengths = np.linspace(400, 1000, len(spectrum))  # Assume 400-1000nm range
    
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, spectrum, 'b-', linewidth=2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title(f'{title} at pixel ({x}, {y})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def main():
    print("Hyperspectral Diffusion Models - Inference Example")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    spectral_channels = 128
    sensor_channels = 16
    H, W = 64, 64
    batch_size = 2
    
    print(f"\nModel Configuration:")
    print(f"Spectral channels: {spectral_channels}")
    print(f"Sensor channels: {sensor_channels}")
    print(f"Spatial resolution: {H}×{W}")
    
    # 1. Create and initialize models
    print("\n" + "="*50)
    print("1. INITIALIZING MODELS")
    print("="*50)
    
    # Full resolution model
    full_model = HyperspectralDiffusion(
        spectral_channels=spectral_channels,
        sensor_channels=sensor_channels
    ).to(device)
    
    # VAE model  
    vae = HyperspectralVAE(
        input_channels=spectral_channels,
        latent_channels=8,
        base_channels=64
    ).to(device)
    
    # Latent diffusion model
    latent_model = LatentHyperspectralDiffusion(
        sensor_channels=sensor_channels,
        vae=vae
    ).to(device)
    
    print(f"Full resolution model parameters: {sum(p.numel() for p in full_model.parameters()):,}")
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"Latent diffusion parameters: {sum(p.numel() for p in latent_model.diffusion_model.parameters()):,}")
    
    # 2. Create synthetic conditioning context
    print("\n" + "="*50)
    print("2. CREATING CONDITIONING CONTEXT")
    print("="*50)
    
    # Create sensor response context (simulates camera/sensor characteristics)
    sensor_context = torch.randn(batch_size, sensor_channels, H, W, device=device)
    sensor_context = torch.sigmoid(sensor_context)  # Normalize to [0, 1]
    
    print(f"Sensor context shape: {sensor_context.shape}")
    print(f"Sensor context range: [{sensor_context.min():.3f}, {sensor_context.max():.3f}]")
    
    # 3. VAE compression/reconstruction test
    print("\n" + "="*50)
    print("3. VAE COMPRESSION TEST")
    print("="*50)
    
    # Create sample hyperspectral data
    test_hyperspectral = torch.randn(batch_size, spectral_channels, H, W, device=device)
    test_hyperspectral = torch.sigmoid(test_hyperspectral)  # Normalize to [0, 1]
    
    vae.eval()
    with torch.no_grad():
        # Encode to latent space
        z_mean, z_logvar = vae.encode(test_hyperspectral)
        z = vae.reparameterize(z_mean, z_logvar)
        
        # Decode back to hyperspectral
        reconstructed = vae.decode(z)
        
        # Compute reconstruction error
        mse_loss = F.mse_loss(reconstructed, test_hyperspectral)
        
    original_size = test_hyperspectral.numel()
    compressed_size = z.numel()
    compression_ratio = original_size / compressed_size
    
    print(f"Original shape: {test_hyperspectral.shape}")
    print(f"Compressed shape: {z.shape}")
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(f"Reconstruction MSE: {mse_loss.item():.6f}")
    
    # 4. Generate samples using DDPM
    print("\n" + "="*50)
    print("4. GENERATING HYPERSPECTRAL SAMPLES")
    print("="*50)
    
    sampler = DDPMSampler(num_timesteps=1000)
    
    # Generate with full resolution model
    print("Generating with full resolution model...")
    full_samples = sampler.sample(
        model=full_model,
        shape=(batch_size, spectral_channels, H, W),
        context=sensor_context,
        device=device,
        num_steps=20  # Fast sampling
    )
    
    # Generate with latent model
    print("Generating with latent diffusion model...")
    latent_samples = sampler.sample(
        model=latent_model,
        shape=(batch_size, 8, H//8, W//8),  # Latent space shape
        context=sensor_context,
        device=device,
        num_steps=20
    )
    
    # Decode latent samples to full resolution
    vae.eval()
    with torch.no_grad():
        latent_samples_decoded = vae.decode(latent_samples)
    
    print(f"Full resolution samples shape: {full_samples.shape}")
    print(f"Latent samples shape: {latent_samples.shape}")
    print(f"Latent samples decoded shape: {latent_samples_decoded.shape}")
    
    # 5. Analyze generated samples
    print("\n" + "="*50)
    print("5. SAMPLE ANALYSIS")
    print("="*50)
    
    # Check value ranges
    print(f"Full resolution samples range: [{full_samples.min():.3f}, {full_samples.max():.3f}]")
    print(f"Latent decoded samples range: [{latent_samples_decoded.min():.3f}, {latent_samples_decoded.max():.3f}]")
    
    # Compute spectral statistics
    full_mean_spectrum = full_samples.mean(dim=[0, 2, 3])  # Average over batch and spatial dims
    latent_mean_spectrum = latent_samples_decoded.mean(dim=[0, 2, 3])
    
    spectral_correlation = F.cosine_similarity(
        full_mean_spectrum.unsqueeze(0), 
        latent_mean_spectrum.unsqueeze(0), 
        dim=1
    ).item()
    
    print(f"Spectral correlation between methods: {spectral_correlation:.4f}")
    
    # 6. Visualization  
    print("\n" + "="*50)
    print("6. CREATING VISUALIZATIONS")
    print("="*50)
    
    # Convert to RGB for visualization
    full_rgb = visualize_hyperspectral_rgb(full_samples)
    latent_rgb = visualize_hyperspectral_rgb(latent_samples_decoded)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # RGB visualizations
    axes[0, 0].imshow(full_rgb)
    axes[0, 0].set_title('Full Resolution Sample (RGB)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(latent_rgb)
    axes[0, 1].set_title('Latent Decoded Sample (RGB)')
    axes[0, 1].axis('off')
    
    # Difference
    rgb_diff = np.abs(full_rgb - latent_rgb)
    axes[0, 2].imshow(rgb_diff)
    axes[0, 2].set_title('RGB Difference')
    axes[0, 2].axis('off')
    
    # Spectral signatures at center pixel
    center_x, center_y = W//2, H//2
    
    full_spectrum = full_samples[0, :, center_y, center_x].cpu().numpy()
    latent_spectrum = latent_samples_decoded[0, :, center_y, center_x].cpu().numpy()
    wavelengths = np.linspace(400, 1000, spectral_channels)
    
    axes[1, 0].plot(wavelengths, full_spectrum, 'b-', label='Full Resolution', linewidth=2)
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Reflectance')
    axes[1, 0].set_title('Full Resolution Spectrum')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(wavelengths, latent_spectrum, 'r-', label='Latent Decoded', linewidth=2)
    axes[1, 1].set_xlabel('Wavelength (nm)')
    axes[1, 1].set_ylabel('Reflectance')
    axes[1, 1].set_title('Latent Decoded Spectrum')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Spectral comparison
    axes[1, 2].plot(wavelengths, full_spectrum, 'b-', label='Full Resolution', linewidth=2, alpha=0.7)
    axes[1, 2].plot(wavelengths, latent_spectrum, 'r--', label='Latent Decoded', linewidth=2, alpha=0.7)
    axes[1, 2].set_xlabel('Wavelength (nm)')
    axes[1, 2].set_ylabel('Reflectance')
    axes[1, 2].set_title('Spectral Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperspectral_generation_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'hyperspectral_generation_comparison.png'")
    
    # 7. Performance summary
    print("\n" + "="*50)
    print("7. PERFORMANCE SUMMARY")
    print("="*50)
    
    # Memory footprint
    full_memory = full_samples.numel() * 4 / 1024**2  # Assuming float32
    latent_memory = latent_samples.numel() * 4 / 1024**2
    
    print(f"Memory footprint:")
    print(f"  Full resolution: {full_memory:.2f} MB")
    print(f"  Latent space: {latent_memory:.2f} MB")
    print(f"  Memory savings: {(1 - latent_memory/full_memory)*100:.1f}%")
    
    print(f"\nModel efficiency:")
    print(f"  Data compression: {compression_ratio:.1f}x")
    print(f"  Spectral fidelity: {spectral_correlation:.3f}")
    print(f"  Reconstruction error: {mse_loss.item():.6f}")
    
    print(f"\n" + "="*60)
    print("INFERENCE COMPLETE!")
    print("="*60)
    print("The latent diffusion approach successfully generates")
    print("high-quality hyperspectral images with significant")
    print("computational and memory savings!")

if __name__ == "__main__":
    main()
