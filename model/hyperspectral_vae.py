import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ConvBlock2D
import math


class HyperspectralVAE(nn.Module):
    """
    Variational Autoencoder for hyperspectral images.
    
    Compresses hyperspectral images from (B, L, H, W) to (B, 8, H/8, W/8)
    where L is the number of spectral channels (e.g., 128-200 bands).
    
    This follows a similar architecture to Stable Diffusion's VAE but is adapted
    for hyperspectral data with many more input channels.
    
    Architecture:
        Encoder: (B, L, H, W) → (B, 8, H/8, W/8) 
        - Progressive downsampling: H,W → H/2,W/2 → H/4,W/4 → H/8,W/8
        - Channel evolution: L → 128 → 256 → 512 → 16 (8 for mean + 8 for logvar)
        
        Decoder: (B, 8, H/8, W/8) → (B, L, H, W)
        - Progressive upsampling: H/8,W/8 → H/4,W/4 → H/2,W/2 → H,W
        - Channel evolution: 8 → 512 → 256 → 128 → L
    """
    
    def __init__(self, spectral_channels, latent_channels=8, base_channels=128):
        """
        Args:
            spectral_channels (int): Number of input spectral channels (L)
            latent_channels (int): Number of latent space channels (default: 8)
            base_channels (int): Base number of channels for the network (default: 128)
        """
        super(HyperspectralVAE, self).__init__()
        
        self.spectral_channels = spectral_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        
        # Encoder: (B, L, H, W) → (B, 8, H/8, W/8)
        self.encoder = HyperspectralEncoder(spectral_channels, latent_channels, base_channels)
        
        # Decoder: (B, 8, H/8, W/8) → (B, L, H, W)
        self.decoder = HyperspectralDecoder(latent_channels, spectral_channels, base_channels)
        
        # Quantization layer for stable training (optional)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
    
    def encode(self, x):
        """
        Encode hyperspectral image to latent representation.
        
        Args:
            x: Input tensor [B, L, H, W]
        
        Returns:
            mean: Mean of latent distribution [B, 8, H/8, W/8]
            logvar: Log variance of latent distribution [B, 8, H/8, W/8]
        """
        # Encoder forward pass
        h = self.encoder(x)  # [B, L, H, W] → [B, 16, H/8, W/8]
        
        # Quantization layer
        moments = self.quant_conv(h)  # [B, 16, H/8, W/8]
        
        # Split into mean and log variance
        mean, logvar = torch.chunk(moments, 2, dim=1)  # Each [B, 8, H/8, W/8]
        
        return mean, logvar
    
    def decode(self, z):
        """
        Decode latent representation back to hyperspectral image.
        
        Args:
            z: Latent tensor [B, 8, H/8, W/8]
        
        Returns:
            Reconstructed hyperspectral image [B, L, H, W]
        """
        # Post-quantization convolution
        z = self.post_quant_conv(z)  # [B, 8, H/8, W/8]
        
        # Decoder forward pass
        x_recon = self.decoder(z)  # [B, 8, H/8, W/8] → [B, L, H, W]
        
        return x_recon
    
    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick for VAE training.
        
        Args:
            mean: Mean tensor [B, 8, H/8, W/8]
            logvar: Log variance tensor [B, 8, H/8, W/8]
        
        Returns:
            Sampled latent tensor [B, 8, H/8, W/8]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean
    
    def forward(self, x, sample=True):
        """
        Full VAE forward pass.
        
        Args:
            x: Input hyperspectral image [B, L, H, W]
            sample: Whether to sample from latent distribution (default: True)
        
        Returns:
            x_recon: Reconstructed image [B, L, H, W]
            mean: Latent mean [B, 8, H/8, W/8]
            logvar: Latent log variance [B, 8, H/8, W/8]
        """
        # Encode
        mean, logvar = self.encode(x)
        
        # Sample from latent distribution
        if sample:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        
        # Decode
        x_recon = self.decode(z)
        
        return x_recon, mean, logvar


class HyperspectralEncoder(nn.Module):
    """
    Encoder network for hyperspectral VAE.
    
    Progressive downsampling with spectral channel compression:
    (B, L, H, W) → (B, 128, H, W) → (B, 256, H/2, W/2) → (B, 512, H/4, W/4) → (B, 16, H/8, W/8)
    """
    
    def __init__(self, spectral_channels, latent_channels, base_channels):
        super(HyperspectralEncoder, self).__init__()
        
        # Input projection: Reduce spectral channels to manageable number
        self.input_proj = nn.Sequential(
            nn.Conv2d(spectral_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        
        # Down block 1: H,W → H/2,W/2
        self.down1 = nn.Sequential(
            ConvBlock2D(base_channels, base_channels),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        
        # Down block 2: H/2,W/2 → H/4,W/4
        self.down2 = nn.Sequential(
            ConvBlock2D(base_channels * 2, base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )
        
        # Down block 3: H/4,W/4 → H/8,W/8
        self.down3 = nn.Sequential(
            ConvBlock2D(base_channels * 4, base_channels * 4),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )
        
        # Output projection: Map to latent space (2x for mean and logvar)
        self.output_proj = nn.Sequential(
            ConvBlock2D(base_channels * 4, base_channels * 2),
            nn.Conv2d(base_channels * 2, latent_channels * 2, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, L, H, W]
        
        Returns:
            Output tensor [B, 16, H/8, W/8] (for latent_channels=8)
        """
        x = self.input_proj(x)     # [B, L, H, W] → [B, 128, H, W]
        x = self.down1(x)          # [B, 128, H, W] → [B, 256, H/2, W/2]
        x = self.down2(x)          # [B, 256, H/2, W/2] → [B, 512, H/4, W/4]
        x = self.down3(x)          # [B, 512, H/4, W/4] → [B, 512, H/8, W/8]
        x = self.output_proj(x)    # [B, 512, H/8, W/8] → [B, 16, H/8, W/8]
        
        return x


class HyperspectralDecoder(nn.Module):
    """
    Decoder network for hyperspectral VAE.
    
    Progressive upsampling with spectral channel expansion:
    (B, 8, H/8, W/8) → (B, 512, H/8, W/8) → (B, 256, H/4, W/4) → (B, 128, H/2, W/2) → (B, L, H, W)
    """
    
    def __init__(self, latent_channels, spectral_channels, base_channels):
        super(HyperspectralDecoder, self).__init__()
        
        # Input projection: Map from latent space
        self.input_proj = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )
        
        # Up block 1: H/8,W/8 → H/4,W/4
        self.up1 = nn.Sequential(
            ConvBlock2D(base_channels * 4, base_channels * 4),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        
        # Up block 2: H/4,W/4 → H/2,W/2
        self.up2 = nn.Sequential(
            ConvBlock2D(base_channels * 2, base_channels * 2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        
        # Up block 3: H/2,W/2 → H,W
        self.up3 = nn.Sequential(
            ConvBlock2D(base_channels, base_channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        
        # Output projection: Map to spectral channels
        self.output_proj = nn.Sequential(
            ConvBlock2D(base_channels, base_channels // 2),
            nn.Conv2d(base_channels // 2, spectral_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 8, H/8, W/8]
        
        Returns:
            Output tensor [B, L, H, W]
        """
        x = self.input_proj(x)     # [B, 8, H/8, W/8] → [B, 512, H/8, W/8]
        x = self.up1(x)           # [B, 512, H/8, W/8] → [B, 256, H/4, W/4]
        x = self.up2(x)           # [B, 256, H/4, W/4] → [B, 128, H/2, W/2]
        x = self.up3(x)           # [B, 128, H/2, W/2] → [B, 128, H, W]
        x = self.output_proj(x)   # [B, 128, H, W] → [B, L, H, W]
        
        return x


def vae_loss(x_recon, x_orig, mean, logvar, kl_weight=1e-6):
    """
    VAE loss function combining reconstruction loss and KL divergence.
    
    Args:
        x_recon: Reconstructed images [B, L, H, W]
        x_orig: Original images [B, L, H, W]
        mean: Latent means [B, 8, H/8, W/8]
        logvar: Latent log variances [B, 8, H/8, W/8]
        kl_weight: Weight for KL divergence term (default: 1e-6)
    
    Returns:
        total_loss: Combined VAE loss
        recon_loss: Reconstruction loss (L2)
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (L2)
    recon_loss = F.mse_loss(x_recon, x_orig, reduction='mean')
    
    # KL divergence loss: KL(q(z|x) || p(z)) where p(z) = N(0, I)
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / mean.numel()
    
    # Total VAE loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # Test the hyperspectral VAE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing HyperspectralVAE on {device}")
    
    # Configuration
    batch_size = 4
    spectral_channels = 128  # Number of hyperspectral bands
    height, width = 64, 64
    latent_channels = 8
    base_channels = 128
    
    # Create model
    vae = HyperspectralVAE(spectral_channels, latent_channels, base_channels).to(device)
    
    # Create test input
    x = torch.randn(batch_size, spectral_channels, height, width).to(device)
    
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters() if p.requires_grad):,}")
    print(f"Input shape: {x.shape}")
    
    # Test encoding
    with torch.no_grad():
        mean, logvar = vae.encode(x)
        print(f"Latent mean shape: {mean.shape}")
        print(f"Latent logvar shape: {logvar.shape}")
        
        # Test reparameterization
        z = vae.reparameterize(mean, logvar)
        print(f"Sampled latent shape: {z.shape}")
        
        # Test decoding
        x_recon = vae.decode(z)
        print(f"Reconstructed shape: {x_recon.shape}")
        
        # Test full forward pass
        x_recon_full, mean_full, logvar_full = vae(x)
        print(f"Full forward - reconstructed shape: {x_recon_full.shape}")
        
        # Test loss computation
        total_loss, recon_loss, kl_loss = vae_loss(x_recon_full, x, mean_full, logvar_full)
        print(f"VAE Loss - Total: {total_loss:.6f}, Reconstruction: {recon_loss:.6f}, KL: {kl_loss:.6f}")
        
        # Test compression ratio
        original_size = x.numel()
        compressed_size = z.numel()
        compression_ratio = original_size / compressed_size
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Original size: {original_size:,} elements")
        print(f"Compressed size: {compressed_size:,} elements")
    
    # Test different input sizes
    print("\nTesting different input sizes:")
    test_sizes = [(32, 32), (128, 128), (256, 256)]
    
    for h, w in test_sizes:
        try:
            x_test = torch.randn(1, spectral_channels, h, w).to(device)
            with torch.no_grad():
                x_recon_test, _, _ = vae(x_test)
            print(f"Size {h}x{w}: {x_test.shape} → {x_recon_test.shape} ✓")
        except Exception as e:
            print(f"Size {h}x{w}: Failed - {e}")
    
    print("\nHyperspectralVAE test completed successfully!")
