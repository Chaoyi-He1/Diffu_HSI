import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from hyperspectral_vae import HyperspectralVAE
import math


class LatentU2NetHyperspectral(nn.Module):
    """
    Latent space U2Net diffusion model for hyperspectral images.
    
    This model operates in the compressed latent space created by the HyperspectralVAE,
    making it much more memory and computationally efficient.
    
    Input/Output Format:
    - Input (z_t): Noisy latent representation [B, 8, H/8, W/8]
    - Context: Sensor response [B, S, H, W] (full resolution)
    - Time: Diffusion timestep [B]
    - Output: Predicted noise in latent space [B, 8, H/8, W/8]
    
    The key advantage is that we work with 8 latent channels instead of 128+ spectral channels,
    and at 8x8 lower spatial resolution, resulting in ~1000x less data to process.
    """
    
    def __init__(self, sensor_channels, latent_channels=8, base_channels=64):
        """
        Args:
            sensor_channels (int): Number of sensor response channels (S)
            latent_channels (int): Number of latent channels from VAE (default: 8)
            base_channels (int): Base number of channels for the network (default: 64)
        """
        super(LatentU2NetHyperspectral, self).__init__()
        
        self.sensor_channels = sensor_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        
        # Time embedding [B] → [B, Base_C*8]
        time_dim = base_channels * 8
        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Context encoder for sensor response conditioning
        # Since we work at 8x lower resolution, we need to downsample the context appropriately
        self.context_encoder = nn.Sequential(
            # First downsample to H/2, W/2
            nn.Conv2d(sensor_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            # Then to H/4, W/4
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
            # Finally to H/8, W/8 to match latent resolution
            nn.Conv2d(base_channels * 2, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.GELU(),
        )
        
        # Input projection from latent channels to base channels
        self.input_proj = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder Path: Work at the compressed resolution
        # Since we're already at H/8, W/8, we can do fewer downsampling steps
        
        # Stage 1: Process at H/8, W/8
        self.stage1 = LatentU2NetBlock(base_channels, base_channels, base_channels,
                                      context_dim=base_channels * 8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/8, W/8 → H/16, W/16
        
        # Stage 2: Process at H/16, W/16
        self.stage2 = LatentU2NetBlock(base_channels, base_channels * 2, base_channels * 2,
                                      context_dim=base_channels * 8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/16, W/16 → H/32, W/32
        
        # Bridge: Process at H/32, W/32
        bridge_channels = base_channels * 4
        self.bridge = ConvBlock2D(base_channels * 2, bridge_channels)
        self.bridge_norm = nn.LayerNorm(bridge_channels)
        self.bridge_attn = BasicTransformerBlock(
            dim=bridge_channels,
            context_dim=base_channels * 8,  # Add context_dim parameter
            n_heads=8,
            d_head=bridge_channels//8,
            gated_ff=True,
            dropout=0.0
        )
        
        # Decoder Path
        # Stage 3: H/32, W/32 → H/16, W/16
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.stage3 = LatentU2NetBlock(bridge_channels + base_channels * 2, 
                                      base_channels * 2, base_channels * 2,
                                      context_dim=base_channels * 8)
        
        # Stage 4: H/16, W/16 → H/8, W/8
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.stage4 = LatentU2NetBlock(base_channels * 2 + base_channels, 
                                      base_channels, base_channels,
                                      context_dim=base_channels * 8)
        
        # Final projection to latent channels
        self.final = nn.Conv2d(base_channels, latent_channels, kernel_size=1)

    def forward(self, z_t, t, context):
        """
        Forward pass through latent U2Net
        
        Args:
            z_t: Noisy latent tensor [B, 8, 64, 64] (VAE latent space)
            t: Timestep [B]
            context: Sensor response context [B, S, 512, 512] (original full resolution)
            
        Returns:
            Predicted noise in latent space [B, 8, 64, 64] (same as VAE latent resolution)
        """
        B, _, H_latent, W_latent = z_t.shape  # H_latent = 64, W_latent = 64 (VAE latent)
        
        # Time embedding
        t_emb = self.time_embedding(t)  # [B] → [B, Base_C*8]
        
        # Context encoding - downsample from original 512x512 to VAE latent 64x64
        # The context_encoder downsamples 512x512 → 64x64 to match VAE latent resolution  
        context_encoded = self.context_encoder(context)  # [B, S, 512, 512] → [B, Base_C*8, 64, 64]
        
        # Helper function to create context for different U2Net block resolutions
        def create_context_for_resolution(target_h, target_w):
            # Downsample encoded context to target resolution
            orig_h, orig_w = context_encoded.shape[2], context_encoded.shape[3]
            if target_h != orig_h or target_w != orig_w:
                context_down = F.interpolate(context_encoded, size=(target_h, target_w), 
                                           mode='bilinear', align_corners=False)
            else:
                context_down = context_encoded
            
            # Convert to attention format [B, target_h*target_w, Base_C*8]
            context_flat = context_down.view(B, self.base_channels * 8, target_h * target_w).permute(0, 2, 1)
            
            # Add time embedding
            t_emb_expanded = t_emb[:, None, :].expand(-1, target_h * target_w, -1)
            return context_flat + t_emb_expanded
        
        # Input projection
        x = self.input_proj(z_t)  # [B, 8, H/8, W/8] → [B, Base_C, H/8, W/8]
        
        # Encoder path
        # Stage 1: Process at latent resolution H/8, W/8 -> context should match spatial resolution
        context_full = create_context_for_resolution(H_latent, W_latent)  
        x1 = self.stage1(x, context_full)  # [B, Base_C, H/8, W/8] → [B, Base_C, H/8, W/8]
        x = self.pool1(x1)  # [B, Base_C, H/8, W/8] → [B, Base_C, H/16, W/16]
        
        # Stage 2: Process at half latent resolution H/16, W/16
        context_half = create_context_for_resolution(H_latent//2, W_latent//2)
        x2 = self.stage2(x, context_half)  # [B, Base_C, H/16, W/16] → [B, Base_C*2, H/16, W/16]
        x = self.pool2(x2)  # [B, Base_C*2, H/16, W/16] → [B, Base_C*2, H/32, W/32]
        
        # Bridge: Process at quarter latent resolution H/32, W/32
        x = self.bridge(x)  # [B, Base_C*2, H/32, W/32] → [B, Base_C*4, H/32, W/32]
        
        bridge_h, bridge_w = H_latent//4, W_latent//4
        context_bridge = create_context_for_resolution(bridge_h, bridge_w)
        
        # Apply attention at bridge level
        x_flat = x.view(B, self.base_channels * 4, bridge_h * bridge_w).permute(0, 2, 1)
        x_flat = self.bridge_attn(x_flat, context_bridge)
        x = x_flat.permute(0, 2, 1).view(B, self.base_channels * 4, bridge_h, bridge_w)
        
        # Decoder path
        # Stage 3: H/32, W/32 → H/16, W/16  
        x = self.up1(x)  # [B, Base_C*4, H/32, W/32] → [B, Base_C*4, H/16, W/16]
        x = torch.cat([x, x2], dim=1)  # [B, Base_C*4+Base_C*2, H/16, W/16]
        x = self.stage3(x, context_half)  # → [B, Base_C*2, H/16, W/16]
        
        # Stage 4: H/16, W/16 → H/8, W/8
        x = self.up2(x)  # [B, Base_C*2, H/16, W/16] → [B, Base_C*2, H/8, W/8]
        x = torch.cat([x, x1], dim=1)  # [B, Base_C*2+Base_C, H/8, W/8]
        x = self.stage4(x, context_full)  # → [B, Base_C, H/8, W/8]
        
        # Final output projection
        x = self.final(x)  # [B, Base_C, H/8, W/8] → [B, 8, H/8, W/8]
        
        return x


class LatentU2NetBlock(nn.Module):
    """
    Simplified U2Net block for latent space processing.
    Since we're working in latent space, we can use simpler blocks.
    """
    def __init__(self, in_channels, mid_channels, out_channels, context_dim):
        super(LatentU2NetBlock, self).__init__()
        
        # Simple 3-stage nested structure (reduced complexity for latent space)
        self.stage1 = ConvBlock2D(in_channels, mid_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm1 = nn.LayerNorm(mid_channels)
        self.context_proj1 = nn.Linear(context_dim, mid_channels)
        self.attn1 = BasicTransformerBlock(
            dim=mid_channels, n_heads=4, d_head=mid_channels//4, gated_ff=True
        )
        
        self.stage2 = ConvBlock2D(mid_channels, mid_channels)
        self.norm2 = nn.LayerNorm(mid_channels)
        self.context_proj2 = nn.Linear(context_dim, mid_channels)
        self.attn2 = BasicTransformerBlock(
            dim=mid_channels, n_heads=4, d_head=mid_channels//4, gated_ff=True
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.stage3 = ConvBlock2D(mid_channels * 2, out_channels)
        self.norm3 = nn.LayerNorm(out_channels)
        self.context_proj3 = nn.Linear(context_dim, out_channels)
        self.attn3 = BasicTransformerBlock(
            dim=out_channels, n_heads=4, d_head=out_channels//4, gated_ff=True
        )

    def forward(self, x, context=None):
        '''
        Forward pass for the LatentU2NetBlock.
            x: Input tensor of shape [B, C, H, W]
            context: Context tensor of shape [B, HW, C]
            return: Output tensor of shape [B, C, H, W]
        '''
        def to_attention_format(x):
            B, C, H, W = x.shape
            return x.view(B, C, H*W).permute(0, 2, 1)
        
        def from_attention_format(x, H, W):
            B, HW, C = x.shape
            return x.permute(0, 2, 1).view(B, C, H, W)
        
        def downsample_context(context, target_h, target_w):
            if context is None:
                return None
            B, orig_hw, C = context.shape
            orig_h = orig_w = int(math.sqrt(orig_hw))
            context_spatial = context.permute(0, 2, 1).view(B, C, orig_h, orig_w)
            context_down = F.interpolate(context_spatial, size=(target_h, target_w), 
                                       mode='bilinear', align_corners=True)
            return context_down.view(B, C, target_h * target_w).permute(0, 2, 1)
        
        _, _, orig_H, orig_W = x.shape
        
        # Stage 1: Process and downsample
        x1 = self.stage1(x)  # [B, in_ch, H, W] → [B, mid_ch, H, W]
        x1_att = to_attention_format(x1)
        if context is not None:
            context_proj = self.context_proj1(context)
            x1_att = self.attn1(x1_att, context_proj)
        else:
            x1_att = self.attn1(x1_att, None)
        x1 = from_attention_format(x1_att, orig_H, orig_W)
        x = self.pool1(x1)  # [B, mid_ch, H, W] → [B, mid_ch, H/2, W/2]
        
        # Stage 2: Process at lower resolution
        x = self.stage2(x)  # [B, mid_ch, H/2, W/2] → [B, mid_ch, H/2, W/2]
        x_att = to_attention_format(x)
        if context is not None:
            context_down = downsample_context(context, orig_H//2, orig_W//2)
            context_proj = self.context_proj2(context_down)
            x_att = self.attn2(x_att, context_proj)
        else:
            x_att = self.attn2(x_att, None)
        x = from_attention_format(x_att, orig_H//2, orig_W//2)
        
        # Stage 3: Upsample and combine
        x = self.up1(x)  # [B, mid_ch, H/2, W/2] → [B, mid_ch, H, W]
        x = torch.cat([x, x1], dim=1)  # [B, mid_ch*2, H, W]
        x = self.stage3(x)  # [B, mid_ch*2, H, W] → [B, out_ch, H, W]
        x_att = to_attention_format(x)
        if context is not None:
            context_proj = self.context_proj3(context)
            x_att = self.attn3(x_att, context_proj)
        else:
            x_att = self.attn3(x_att, None)
        x = from_attention_format(x_att, orig_H, orig_W)
        
        return x


class LatentHyperspectralDiffusion(nn.Module):
    """
    Complete latent diffusion system combining VAE and U2Net.
    
    This class manages both the VAE for encoding/decoding and the diffusion model
    for processing in latent space.
    """
    
    def __init__(self, spectral_channels, sensor_channels, latent_channels=8, 
                 base_channels=64, vae_base_channels=128, vae_checkpoint_path=None):
        """
        Args:
            spectral_channels (int): Number of spectral channels in hyperspectral images
            sensor_channels (int): Number of sensor response channels
            latent_channels (int): Number of latent space channels (default: 8)
            base_channels (int): Base channels for diffusion model (default: 64)
            vae_base_channels (int): Base channels for VAE (default: 128)
            vae_checkpoint_path (str, optional): Path to pre-trained VAE checkpoint
        """
        super(LatentHyperspectralDiffusion, self).__init__()
        
        # VAE for encoding/decoding hyperspectral images
        self.vae = HyperspectralVAE(spectral_channels, latent_channels, vae_base_channels)
        
        # Load pre-trained VAE if checkpoint path is provided
        if vae_checkpoint_path is not None:
            self.load_vae_checkpoint(vae_checkpoint_path)
        
        # U2Net diffusion model operating in latent space
        self.diffusion_model = LatentU2NetHyperspectral(sensor_channels, latent_channels, base_channels)
        
        # Freeze VAE during diffusion training (optional)
        self.freeze_vae()
    
    def freeze_vae(self):
        """Freeze VAE parameters during diffusion training."""
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def unfreeze_vae(self):
        """Unfreeze VAE parameters for joint training."""
        for param in self.vae.parameters():
            param.requires_grad = True
    
    def load_vae_checkpoint(self, checkpoint_path):
        """
        Load pre-trained VAE from checkpoint.
        
        Args:
            checkpoint_path (str): Path to the VAE checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Standard checkpoint with model_state_dict
            self.vae.load_state_dict(checkpoint['vae_model'])
            print(f"Loaded VAE from checkpoint: {checkpoint_path}")
            if 'epoch' in checkpoint:
                print(f"  - Checkpoint epoch: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                print(f"  - Checkpoint loss: {checkpoint['loss']:.4f}")

        except FileNotFoundError:
            raise FileNotFoundError(f"VAE checkpoint not found at: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load VAE checkpoint: {str(e)}")
    
    def save_vae_checkpoint(self, checkpoint_path, epoch=None, loss=None):
        """
        Save current VAE state to checkpoint.
        
        Args:
            checkpoint_path (str): Path to save the checkpoint
            epoch (int, optional): Current training epoch
            loss (float, optional): Current loss value
        """
        checkpoint = {
            'vae_model': self.vae.state_dict(),
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved VAE checkpoint to: {checkpoint_path}")
    
    def save_diffusion_checkpoint(self, checkpoint_path, epoch=None, loss=None):
        """
        Save current diffusion model state to checkpoint.
        
        Args:
            checkpoint_path (str): Path to save the checkpoint
            epoch (int, optional): Current training epoch
            loss (float, optional): Current loss value
        """
        checkpoint = {
            'diffusion_model': self.diffusion_model.state_dict(),
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved diffusion model checkpoint to: {checkpoint_path}")
    
    def save_full_checkpoint(self, checkpoint_path, epoch=None, loss=None):
        """
        Save complete model state (both VAE and diffusion model) to checkpoint.
        
        Args:
            checkpoint_path (str): Path to save the checkpoint
            epoch (int, optional): Current training epoch
            loss (float, optional): Current loss value
        """
        checkpoint = {
            'vae_model': self.vae.state_dict(),
            'diffusion_model': self.diffusion_model.state_dict(),
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved complete model checkpoint to: {checkpoint_path}")
    
    def load_diffusion_checkpoint(self, checkpoint_path):
        """
        Load pre-trained diffusion model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to the diffusion model checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            self.diffusion_model.load_state_dict(checkpoint['diffusion_model'])
            print(f"Loaded diffusion model from checkpoint: {checkpoint_path}")
            if 'epoch' in checkpoint:
                print(f"  - Checkpoint epoch: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                print(f"  - Checkpoint loss: {checkpoint['loss']:.4f}")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Diffusion checkpoint not found at: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load diffusion checkpoint: {str(e)}")
    
    def load_full_checkpoint(self, checkpoint_path):
        """
        Load complete model state (both VAE and diffusion model) from checkpoint.
        
        Args:
            checkpoint_path (str): Path to the complete model checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if 'vae_model' in checkpoint and 'diffusion_model' in checkpoint:
                self.vae.load_state_dict(checkpoint['vae_model'])
                self.diffusion_model.load_state_dict(checkpoint['diffusion_model'])
                print(f"Loaded complete model from checkpoint: {checkpoint_path}")
                if 'epoch' in checkpoint:
                    print(f"  - Checkpoint epoch: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    print(f"  - Checkpoint loss: {checkpoint['loss']:.4f}")
            else:
                raise ValueError("Invalid checkpoint format for complete model")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Complete model checkpoint not found at: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load complete model checkpoint: {str(e)}")
    
    def encode_to_latent(self, x, sample=True):
        """
        Encode hyperspectral image to latent space.
        
        Args:
            x: Hyperspectral image [B, L, H, W]
            sample: Whether to sample from latent distribution
        
        Returns:
            z: Latent representation [B, 8, H/8, W/8]
        """
        with torch.no_grad() if not self.vae.training else torch.enable_grad():
            mean, logvar = self.vae.encode(x)
            if sample:
                z = self.vae.reparameterize(mean, logvar)
            else:
                z = mean
        return z
    
    def decode_from_latent(self, z):
        """
        Decode latent representation back to hyperspectral image.
        
        Args:
            z: Latent representation [B, 8, H/8, W/8]
        
        Returns:
            x: Hyperspectral image [B, L, H, W]
        """
        with torch.no_grad() if not self.vae.training else torch.enable_grad():
            x = self.vae.decode(z)
        return x
    
    def forward_generation(self, z_t, t, context):
        """
        Forward pass for generation (inference) using the diffusion model.
        
        Args:
            z_t: Noisy latent tensor [B, 8, H/8, W/8]
            t: Timesteps [B]
            context: Sensor response [B, S, H, W]
        Returns:
            Predicted noise in latent space [B, 8, H/8, W/8]
        """
        predicted_noise = self.diffusion_model(z_t, t, context)
        return predicted_noise
    
    def forward(self, x, t, context):
        """
        Forward pass for training the diffusion model.
        
        Args:
            x: Clean hyperspectral images [B, L, H, W]
            t: Timesteps [B]
            context: Sensor response [B, S, H, W]
        
        Returns:
            Predicted noise in latent space [B, 8, H/8, W/8]
        """
        # Encode to latent space
        z = self.encode_to_latent(x, sample=True)
        
        # Apply diffusion model
        predicted_noise = self.diffusion_model(z, t, context)
        
        return predicted_noise


if __name__ == "__main__":
    # Test the latent diffusion model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing LatentHyperspectralDiffusion on {device}")
    
    # Configuration
    batch_size = 4
    spectral_channels = 128
    sensor_channels = 16
    original_height, original_width = 512, 512  # Original hyperspectral image resolution
    
    # Example: Create model with pre-trained VAE (uncomment to use)
    # vae_checkpoint_path = "/path/to/pretrained_vae.pth"  
    # model = LatentHyperspectralDiffusion(spectral_channels, sensor_channels, 
    #                                    vae_checkpoint_path=vae_checkpoint_path).to(device)
    
    # Create model without pre-trained VAE
    model = LatentHyperspectralDiffusion(spectral_channels, sensor_channels).to(device)
    
    # Example: Load VAE checkpoint after model creation (alternative method)
    # model.load_vae_checkpoint("/path/to/pretrained_vae.pth")
    
    print(f"Model created successfully!")
    print(f"VAE is {'frozen' if not any(p.requires_grad for p in model.vae.parameters()) else 'unfrozen'}")
    
    # Create test data
    # Input HSI: original full resolution (512x512) - will be compressed by VAE to 64x64
    # Context: same original full resolution (512x512) for sensor response
    # VAE will compress HSI 512x512 -> 64x64, then diffusion processes 64x64 -> 8x8
    x = torch.randn(batch_size, spectral_channels, original_height, original_width).to(device)
    context = torch.randn(batch_size, sensor_channels, original_height, original_width).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"VAE parameters: {sum(p.numel() for p in model.vae.parameters()):,}")
    print(f"Diffusion model parameters: {sum(p.numel() for p in model.diffusion_model.parameters()):,}")
    print(f"Input HSI shape: {x.shape}")  # Original full resolution
    print(f"Context shape: {context.shape}")  # Same full resolution
    
    # Test VAE encoding/decoding
    with torch.no_grad():
        # Encode original 512x512 HSI to 64x64 latent space
        z = model.encode_to_latent(x)
        print(f"Latent shape after VAE encoding: {z.shape}")  # Should be [B, 8, 64, 64]
        
        # Decode 64x64 latent back to 512x512 HSI
        x_recon = model.decode_from_latent(z)
        print(f"Reconstructed HSI shape: {x_recon.shape}")  # Should be [B, 128, 512, 512]
        
        # Test diffusion forward pass on latent space
        # Context remains at original resolution (512x512)
        # Diffusion model processes latent z (64x64) and downsamples context as needed
        predicted_noise = model.diffusion_model(z, t, context)
        print(f"Predicted noise shape: {predicted_noise.shape}")  # Should be [B, 8, 8, 8]
        
        # Demonstrate multi-resolution context processing
        latent_h, latent_w = z.shape[2], z.shape[3]  # 64, 64 (VAE latent resolution)
        print(f"\nContext Resolution Processing:")
        print(f"- Original HSI: {x.shape} -> VAE -> Latent: {z.shape}")
        print(f"- Input context: {context.shape} (original full resolution)")
        print(f"- Context encoder output: [B, 512, {latent_h}, {latent_w}] (matches VAE latent resolution)")
        print(f"- Stage 1 & 4 context: [B, {latent_h*latent_w}, 512] (full VAE latent resolution)")
        print(f"- Stage 2 & 3 context: [B, {(latent_h//2)*(latent_w//2)}, 512] (half VAE latent resolution)")  
        print(f"- Bridge context: [B, {(latent_h//4)*(latent_w//4)}, 512] (quarter VAE latent resolution)")
        
        # Calculate compression metrics
        original_elements = x.numel()
        latent_elements = z.numel()
        compression_ratio = original_elements / latent_elements
        memory_savings = (1 - latent_elements/original_elements) * 100
        
        print(f"\nCompression ratio: {compression_ratio:.1f}x")
        print(f"Memory savings: {memory_savings:.1f}%")
        print("LatentHyperspectralDiffusion test completed successfully!")
        
        # Test memory usage comparison
        original_elements = x.numel()
        latent_elements = z.numel()
        compression_ratio = original_elements / latent_elements
        print(f"Compression ratio: {compression_ratio:.1f}x")
        print(f"Memory savings: {(1 - 1/compression_ratio)*100:.1f}%")
        
        # Demonstrate checkpoint saving/loading functionality
        print(f"\nCheckpoint Management Demo:")
        
        # Example: Save VAE checkpoint
        # model.save_vae_checkpoint("vae_checkpoint.pth", epoch=100, loss=0.001)
        
        # Example: Save diffusion model checkpoint  
        # model.save_diffusion_checkpoint("diffusion_checkpoint.pth", epoch=50, loss=0.005)
        
        # Example: Save complete model checkpoint
        # model.save_full_checkpoint("complete_model_checkpoint.pth", epoch=75, loss=0.003)
        
        print("Checkpoint management methods available:")
        print("  - load_vae_checkpoint(path)")
        print("  - save_vae_checkpoint(path, epoch, loss)")
        print("  - load_diffusion_checkpoint(path)")
        print("  - save_diffusion_checkpoint(path, epoch, loss)")
        print("  - load_full_checkpoint(path)")  
        print("  - save_full_checkpoint(path, epoch, loss)")
    
    print("LatentHyperspectralDiffusion test completed successfully!")
