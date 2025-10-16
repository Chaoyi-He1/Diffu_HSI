import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class U2NetBlock2D(nn.Module):
    """
    A nested U-Net block for 2D hyperspectral data processing with transformer attention.
    
    This block implements a mini U-Net structure that processes hyperspectral data through 
    multiple resolution scales while maintaining spatial relationships. Adapted from the 1D 
    version but handles 2D spatial dimensions (H, W) with spectral channels.
    
    Dimension Changes Through the Network:
        Input: [B, Cin, H, W]         # B=batch, Cin=input channels, H=height, W=width
        Stage1: [B, M, H, W]          # M=mid_channels
          ├─► Pool1: [B, M, H/2, W/2] # Downsample spatial dimensions
        Stage2: [B, M, H/2, W/2]
          ├─► Pool2: [B, M, H/4, W/4] # Further downsample
        Stage3: [B, M, H/4, W/4]      # Bridge
          ├─► Up1: [B, M, H/2, W/2]   # Begin upsampling
        Stage4: [B, M, H/2, W/2]      # Concat with Stage2 features: [B, 2M, H/2, W/2] -> [B, M, H/2, W/2]
          ├─► Up2: [B, M, H, W]       # Final upsampling
        Stage5: [B, Cout, H, W]       # Cout=output channels, concatenated with Stage1
    
    Attention Operations:
        - Each stage includes a transformer block that processes data as [B, H*W, C]
        - Input is reshaped from [B, C, H, W] to [B, H*W, C] for attention
        - After attention, data is reshaped back to [B, C, H, W] for convolutions
    
    Skip Connections:
        Stage1 → Stage5: Preserves fine spatial details from highest resolution
        Stage2 → Stage4: Preserves medium-scale spatial features
    """
    def __init__(self, in_channels, mid_channels, out_channels, context_dim):
        """
        Initialize a U2NetBlock2D for hyperspectral diffusion.

        Args:
            in_channels (int): Number of input channels (Cin)
            mid_channels (int): Number of channels in the intermediate layers (M)
                Used throughout the network for feature processing
            out_channels (int): Number of output channels (Cout)
                Used in the final stage to match desired output dimension
            context_dim (int): Dimension of the context vector from parent network
                Used for projecting global context to local features

        Channel Dimension Changes:
            1. Input         : [B, Cin, H, W]       → [B, M, H, W]         (stage1)
            2. Intermediate  : [B, M, H/2, W/2]     → [B, M, H/4, W/4]     (stage2,3)
            3. Skip Concat   : [B, 2M, H/2, W/2]    → [B, M, H/2, W/2]     (stage4)
            4. Final Concat  : [B, 2M, H, W]        → [B, Cout, H, W]      (stage5)

        Attention Dimensions:
            Each attention block processes:
            - Input : [B, C, H, W]    (conv format)
            - Trans : [B, H*W, C]     (attention format)
            - Output: [B, C, H, W]    (back to conv format)
            Where C is either mid_channels or out_channels
        """
        super(U2NetBlock2D, self).__init__()
        
        # Stage 1: Downsampling path [B, Cin, H, W] → [B, M, H, W] → [B, M, H/2, W/2]
        self.stage1 = ConvBlock2D(in_channels, mid_channels)              # [B, Cin, H, W] → [B, M, H, W]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)               # [B, M, H, W] → [B, M, H/2, W/2]
        self.norm1 = nn.LayerNorm(mid_channels)                          # For attention: [B, H*W, M]
        # Context projection from global conditioning to local dimension
        self.context_proj1 = nn.Linear(context_dim, mid_channels)        # Project [B, H*W, context_dim] → [B, H*W, M]
        self.attn1 = BasicTransformerBlock(                              # [B, H*W, M] → [B, H*W, M]
            dim=mid_channels, n_heads=4, 
            d_head=mid_channels//4, gated_ff=True
        )
        
        # Stage 2: Downsampling path [B, M, H/2, W/2] → [B, M, H/2, W/2] → [B, M, H/4, W/4]
        self.stage2 = ConvBlock2D(mid_channels, mid_channels)            # [B, M, H/2, W/2] → [B, M, H/2, W/2]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)               # [B, M, H/2, W/2] → [B, M, H/4, W/4]
        self.norm2 = nn.LayerNorm(mid_channels)                          # For attention: [B, H*W/4, M]
        self.context_proj2 = nn.Linear(context_dim, mid_channels)        # Project [B, H*W, context_dim] → [B, H*W, M]
        self.attn2 = BasicTransformerBlock(                              # [B, H*W/4, M] → [B, H*W/4, M]
            dim=mid_channels, n_heads=4, 
            d_head=mid_channels//4, gated_ff=True
        )
        
        # Stage 3: Bridge [B, M, H/4, W/4] → [B, M, H/4, W/4]
        self.stage3 = ConvBlock2D(mid_channels, mid_channels)            # [B, M, H/4, W/4] → [B, M, H/4, W/4]
        self.norm3 = nn.LayerNorm(mid_channels)                          # For attention: [B, H*W/16, M]
        self.context_proj3 = nn.Linear(context_dim, mid_channels)        # Project [B, H*W, context_dim] → [B, H*W, M]
        self.attn3 = BasicTransformerBlock(                              # [B, H*W/16, M] → [B, H*W/16, M]
            dim=mid_channels, n_heads=4, 
            d_head=mid_channels//4, gated_ff=True
        )
        
        # Stage 4: Upsampling path [B, M, H/4, W/4] → [B, 2M, H/2, W/2] → [B, M, H/2, W/2]
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear',          # [B, M, H/4, W/4] → [B, M, H/2, W/2]
                              align_corners=True)
        self.stage4 = ConvBlock2D(mid_channels * 2, mid_channels)        # [B, 2M, H/2, W/2] → [B, M, H/2, W/2]
        self.norm4 = nn.LayerNorm(mid_channels)                          # For attention: [B, H*W/4, M]
        self.context_proj4 = nn.Linear(context_dim, mid_channels)        # Project [B, H*W, context_dim] → [B, H*W, M]
        self.attn4 = BasicTransformerBlock(                              # [B, H*W/4, M] → [B, H*W/4, M]
            dim=mid_channels, n_heads=4, 
            d_head=mid_channels//4, gated_ff=True
        )
        
        # Stage 5: Upsampling path [B, M, H/2, W/2] → [B, 2M, H, W] → [B, Cout, H, W]
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',          # [B, M, H/2, W/2] → [B, M, H, W]
                              align_corners=True)
        self.stage5 = ConvBlock2D(mid_channels * 2, out_channels)        # [B, 2M, H, W] → [B, Cout, H, W]
        self.norm5 = nn.LayerNorm(out_channels)                          # For attention: [B, H*W, Cout]
        self.context_proj5 = nn.Linear(context_dim, out_channels)        # Project [B, H*W, context_dim] → [B, H*W, Cout]
        self.attn5 = BasicTransformerBlock(                              # [B, H*W, Cout] → [B, H*W, Cout]
            dim=out_channels, n_heads=4, 
            d_head=out_channels//4, gated_ff=True
        )

    def forward(self, x, context=None):
        """
        Forward pass of the U2NetBlock2D.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
                B = batch size
                C = number of channels
                H = height
                W = width
            context (torch.Tensor, optional): Context tensor for conditional attention
                Shape: [B, H*W, context_dim] (will be downsampled as needed)

        Returns:
            torch.Tensor: Output tensor of shape [B, out_channels, H, W]

        Note:
            The block processes data through multiple scales with attention:
            1. Downsampling path: progressively reduces spatial dimensions
            2. Bridge: processes at lowest resolution
            3. Upsampling path: combines features from downsampling path
        """
        # Convert from [B, C, H, W] to [B, H*W, C] for attention
        def to_attention_format(x):
            B, C, H, W = x.shape
            return x.view(B, C, H*W).permute(0, 2, 1)  # [B, C, H*W] -> [B, H*W, C]
        
        def from_attention_format(x, H, W):
            B, HW, C = x.shape
            return x.permute(0, 2, 1).view(B, C, H, W)  # [B, H*W, C] -> [B, C, H, W]
        
        def downsample_context(context, target_h, target_w):
            """Downsample context to match target spatial resolution"""
            if context is None:
                return None
            B, orig_hw, C = context.shape
            orig_h = orig_w = int(math.sqrt(orig_hw))
            
            # Reshape to spatial format and downsample
            context_spatial = context.permute(0, 2, 1).view(B, C, orig_h, orig_w)
            context_down = F.interpolate(context_spatial, size=(target_h, target_w), 
                                       mode='bilinear', align_corners=True)
            # Convert back to attention format, [B, target_h * target_w, C]
            return context_down.view(B, C, target_h * target_w).permute(0, 2, 1)
        
        # Store original dimensions for reconstruction
        _, _, orig_H, orig_W = x.shape
        
        # Encoder path with cross-attention
        # Stage 1: Downsampling path (H,W → H/2,W/2)
        x1 = self.stage1(x)                                    # [B, Cin, H, W]     → [B, M, H, W]
        x1_att = to_attention_format(x1)                       # [B, M, H, W]       → [B, H*W, M]
        if context is not None:
            # Use context at full resolution for stage 1
            context_proj = self.context_proj1(context)         # [B, H*W, M*8]      → [B, H*W, M]
            x1_att = self.attn1(x1_att, context_proj)         # [B, H*W, M]        → [B, H*W, M]
        else:
            x1_att = self.attn1(x1_att, None)                 # Self-attention only
        x1 = from_attention_format(x1_att, orig_H, orig_W)    # [B, H*W, M]        → [B, M, H, W]
        x = self.pool1(x1)                                    # [B, M, H, W]       → [B, M, H/2, W/2]
        
        # Stage 2: Downsampling path (H/2,W/2 → H/4,W/4)
        x2 = self.stage2(x)                                    # [B, M, H/2, W/2]   → [B, M, H/2, W/2]
        x2_att = to_attention_format(x2)                       # [B, M, H/2, W/2]   → [B, H*W/4, M]
        if context is not None:
            # Downsample context for stage 2
            context_down = downsample_context(context, orig_H//2, orig_W//2)
            context_proj = self.context_proj2(context_down)    # [B, H*W/4, M*8]    → [B, H*W/4, M]
            x2_att = self.attn2(x2_att, context_proj)         # [B, H*W/4, M]      → [B, H*W/4, M]
        else:
            x2_att = self.attn2(x2_att, None)                 # Self-attention only
        x2 = from_attention_format(x2_att, orig_H//2, orig_W//2)  # [B, H*W/4, M]   → [B, M, H/2, W/2]
        x = self.pool2(x2)                                    # [B, M, H/2, W/2]   → [B, M, H/4, W/4]
        
        # Stage 3: Bridge (process at lowest resolution)
        x = self.stage3(x)                                    # [B, M, H/4, W/4]   → [B, M, H/4, W/4]
        x_att = to_attention_format(x)                        # [B, M, H/4, W/4]   → [B, H*W/16, M]
        if context is not None:
            # Downsample context for stage 3
            context_down = downsample_context(context, orig_H//4, orig_W//4)
            context_proj = self.context_proj3(context_down)    # [B, H*W/16, M*8]   → [B, H*W/16, M]
            x_att = self.attn3(x_att, context_proj)           # [B, H*W/16, M]     → [B, H*W/16, M]
        else:
            x_att = self.attn3(x_att, None)                   # Self-attention only
        x = from_attention_format(x_att, orig_H//4, orig_W//4) # [B, H*W/16, M]    → [B, M, H/4, W/4]
        
        # Stage 4: Upsampling path with skip connection from Stage 2
        x = self.up1(x)                                       # [B, M, H/4, W/4]   → [B, M, H/2, W/2]
        x = torch.cat([x, x2], dim=1)                         # [B, M, H/2, W/2]   → [B, 2M, H/2, W/2]
        x = self.stage4(x)                                    # [B, 2M, H/2, W/2]  → [B, M, H/2, W/2]
        x_att = to_attention_format(x)                        # [B, M, H/2, W/2]   → [B, H*W/4, M]
        if context is not None:
            # Use downsampled context for stage 4
            context_down = downsample_context(context, orig_H//2, orig_W//2)
            context_proj = self.context_proj4(context_down)    # [B, H*W/4, M*8]    → [B, H*W/4, M]
            x_att = self.attn4(x_att, context_proj)           # [B, H*W/4, M]      → [B, H*W/4, M]
        else:
            x_att = self.attn4(x_att, None)                   # Self-attention only
        x = from_attention_format(x_att, orig_H//2, orig_W//2) # [B, H*W/4, M]     → [B, M, H/2, W/2]
        
        # Stage 5: Upsampling path with skip connection from Stage 1
        x = self.up2(x)                                       # [B, M, H/2, W/2]   → [B, M, H, W]
        x = torch.cat([x, x1], dim=1)                         # [B, M, H, W]       → [B, 2M, H, W]
        x = self.stage5(x)                                    # [B, 2M, H, W]      → [B, Cout, H, W]
        x_att = to_attention_format(x)                        # [B, Cout, H, W]    → [B, H*W, Cout]
        if context is not None:
            # Use full resolution context for stage 5
            context_proj = self.context_proj5(context)         # [B, H*W, M*8]      → [B, H*W, Cout]
            x_att = self.attn5(x_att, context_proj)           # [B, H*W, Cout]     → [B, H*W, Cout]
        else:
            x_att = self.attn5(x_att, None)                   # Self-attention only
        x = from_attention_format(x_att, orig_H, orig_W)      # [B, H*W, Cout]     → [B, Cout, H, W]
        
        return x


class U2NetHyperspectral(nn.Module):
    """
    2D U-Net architecture with nested U-Net blocks and transformer-style attention,
    specifically designed for hyperspectral image conditional diffusion models.
    
    This model follows the Stable Diffusion architecture but is adapted for hyperspectral
    imaging tasks where:
    - x_t has shape (B, L, H, W) where L is the number of spectral channels
    - context has shape (B, S, H, W) where S is the sensor response channels
    
    Architecture Overview (B=batch, L=spectral_channels, H=height, W=width, Base_C=base_channels):
        Input [B, L, H, W]
        ├── Stage1 (UNet Block) ────────────────────────────────────────┐
        │   └── Pool1 [B, Base_C, H, W] → [B, Base_C, H/2, W/2]         │
        ├── Stage2 (UNet Block) ─────────────────────┐                  │
        │   └── Pool2 [B, Base_C*2, H/2, W/2] → [B, Base_C*2, H/4, W/4] │           
        ├── Stage3 (UNet Block) ──────────┐          │                  │
        │   └── Pool3 [B, Base_C*4, H/4, W/4] → [B, Base_C*4, H/8, W/8] │                   
        ├── Bridge [B, Base_C*8, H/8, W/8]│          │                  │
        │   └── Up1                       │          │                  │
        ├── Stage4 (UNet Block) ←─────────┘          │                  │
        │   └── Up2                                  │                  │
        ├── Stage5 (UNet Block) ←────────────────────┘                  │
        │   └── Up3                                                     │
        └── Stage6 (UNet Block) ←───────────────────────────────────────┘
        
    Channel Dimensions (Base_C = base_channels):
        - Stage1: Base_C      (base_channels)
        - Stage2: Base_C*2    (doubles channels)
        - Stage3: Base_C*4    (doubles again)
        - Bridge: Base_C*8    (doubles at bridge)
        - Stage4: Base_C*4    (begins reducing)
        - Stage5: Base_C*2    (reduces further)
        - Stage6: Base_C      (back to base)
    
    Conditioning:
        - Time embedding: Converts timestep to Base_C*8 dimensional vector
        - Context projection: Maps sensor response to Base_C*8 dimensional space
        - Both are combined and used in cross-attention throughout network
    """
    
    def __init__(self, spectral_channels, sensor_channels, base_channels=64):
        """
        Args:
            spectral_channels (int): Number of spectral channels (L) in hyperspectral image
            sensor_channels (int): Number of sensor response channels (S) for conditioning
            base_channels (int): Base number of channels, controls network capacity
                               Each subsequent level multiplies this by 2
        """
        super(U2NetHyperspectral, self).__init__()
        
        self.spectral_channels = spectral_channels
        self.sensor_channels = sensor_channels
        self.base_channels = base_channels
        
        # Time embedding [B] → [B, Base_C*8]
        # Maps diffusion timestep to match context dimension
        time_dim = base_channels * 8
        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),                                    # [B] → [B, Base_C]
            nn.Linear(base_channels, time_dim),                             # [B, Base_C] → [B, Base_C*8]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)                                   # [B, Base_C*8] → [B, Base_C*8]
        )
        
        # Context encoder for sensor response conditioning
        # Processes [B, S, H, W] → [B, H*W, Base_C*8] for attention
        self.context_encoder = nn.Sequential(
            nn.Conv2d(sensor_channels, base_channels, kernel_size=3, padding=1),     # [B, S, H, W] → [B, Base_C, H, W]
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),   # [B, Base_C, H, W] → [B, Base_C*2, H, W]
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 8, kernel_size=1),         # [B, Base_C*2, H, W] → [B, Base_C*8, H, W]
        )
        
        # Input projection from spectral to base channels
        self.input_proj = nn.Conv2d(spectral_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder Path: Progressive downsampling and channel expansion
        # Stage 1: Input level
        self.stage1 = U2NetBlock2D(base_channels, base_channels, base_channels, 
                                  context_dim=base_channels * 8)                    # [B, Base_C, H, W] → [B, Base_C, H, W]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                         # [B, Base_C, H, W] → [B, Base_C, H/2, W/2]
        
        # Stage 2: First downsampling level
        self.stage2 = U2NetBlock2D(base_channels, base_channels * 2, base_channels * 2,
                                  context_dim=base_channels * 8)                    # [B, Base_C, H/2, W/2] → [B, Base_C*2, H/2, W/2]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                         # [B, Base_C*2, H/2, W/2] → [B, Base_C*2, H/4, W/4]
        
        # Stage 3: Second downsampling level
        self.stage3 = U2NetBlock2D(base_channels * 2, base_channels * 4, base_channels * 4,
                                  context_dim=base_channels * 8)                    # [B, Base_C*2, H/4, W/4] → [B, Base_C*4, H/4, W/4]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)                         # [B, Base_C*4, H/4, W/4] → [B, Base_C*4, H/8, W/8]
        
        # Bridge: Bottleneck with maximum channels and attention
        bridge_channels = base_channels * 8
        self.bridge = ConvBlock2D(base_channels * 4, bridge_channels)              # [B, Base_C*4, H/8, W/8] → [B, Base_C*8, H/8, W/8]
        self.bridge_norm = nn.LayerNorm(bridge_channels)                           # For attention: [B, H*W/64, Base_C*8]
        self.bridge_attn = BasicTransformerBlock(
            dim=bridge_channels,                                                   # Processes: [B, H*W/64, Base_C*8]
            n_heads=8,
            d_head=bridge_channels//8,
            gated_ff=True,
            dropout=0.0
        )
        
        # Decoder Path: Progressive upsampling and channel reduction
        # Stage 4: First upsampling level
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # [B, Base_C*8, H/8, W/8] → [B, Base_C*8, H/4, W/4]
        # Concat with stage3: [B, Base_C*8 + Base_C*4, H/4, W/4] = [B, Base_C*12, H/4, W/4]
        self.stage4 = U2NetBlock2D(base_channels * 12, base_channels * 4, base_channels * 4,
                                  context_dim=base_channels * 8)                    # [B, Base_C*12, H/4, W/4] → [B, Base_C*4, H/4, W/4]
        
        # Stage 5: Second upsampling level
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # [B, Base_C*4, H/4, W/4] → [B, Base_C*4, H/2, W/2]
        # Concat with stage2: [B, Base_C*4 + Base_C*2, H/2, W/2] = [B, Base_C*6, H/2, W/2]
        self.stage5 = U2NetBlock2D(base_channels * 6, base_channels * 2, base_channels * 2,
                                  context_dim=base_channels * 8)                    # [B, Base_C*6, H/2, W/2] → [B, Base_C*2, H/2, W/2]
        
        # Stage 6: Final upsampling level
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # [B, Base_C*2, H/2, W/2] → [B, Base_C*2, H, W]
        # Concat with stage1: [B, Base_C*2 + Base_C, H, W] = [B, Base_C*3, H, W]
        self.stage6 = U2NetBlock2D(base_channels * 3, base_channels, base_channels,
                                  context_dim=base_channels * 8)                    # [B, Base_C*3, H, W] → [B, Base_C, H, W]
        
        # Final projection to match spectral channels
        self.final = nn.Conv2d(base_channels, spectral_channels, kernel_size=1)     # [B, Base_C, H, W] → [B, L, H, W]

    def forward(self, x_t, context, t):
        """
        Forward pass of U2NetHyperspectral.
        
        Args:
            x_t: Noisy hyperspectral image tensor of shape [B, L, H, W]
            t: Time embedding tensor of shape [B]
            context: Sensor response tensor of shape [B, S, H, W]
                
        Returns:
            Output tensor of shape [B, L, H, W] - predicted noise or clean image
        """
        B, L, H, W = x_t.shape
        
        # Time embedding
        t_emb = self.time_embedding(t)                            # [B] → [B, Base_C*8]
        
        # Context encoding - process sensor response
        context_encoded = self.context_encoder(context)           # [B, S, H, W] → [B, Base_C*8, H, W]
        
        # Helper function to create context at different resolutions
        def create_context_for_resolution(h, w):
            # Downsample context to target resolution
            context_down = F.interpolate(context_encoded, size=(h, w), 
                                       mode='bilinear', align_corners=True)
            # Convert to attention format [B, h*w, Base_C*8]
            context_flat = context_down.view(B, self.base_channels * 8, h * w).permute(0, 2, 1)
            # Add time embedding
            t_emb_expanded = t_emb[:, None, :].expand(-1, h * w, -1)
            return context_flat + t_emb_expanded
        
        # Input projection from spectral channels to base channels
        x = self.input_proj(x_t)                                  # [B, L, H, W] → [B, Base_C, H, W]
        
        # Encoder path - progressively reduce spatial resolution, increase channels
        context_full = create_context_for_resolution(H, W)        # Context at full resolution
        x1 = self.stage1(x, context_full)                        # [B, Base_C, H, W] → [B, Base_C, H, W]
        x = self.pool1(x1)                                       # [B, Base_C, H, W] → [B, Base_C, H/2, W/2]
        
        context_half = create_context_for_resolution(H//2, W//2)  # Context at half resolution
        x2 = self.stage2(x, context_half)                        # [B, Base_C, H/2, W/2] → [B, Base_C*2, H/2, W/2]
        x = self.pool2(x2)                                       # [B, Base_C*2, H/2, W/2] → [B, Base_C*2, H/4, W/4]
        
        context_quarter = create_context_for_resolution(H//4, W//4)  # Context at quarter resolution
        x3 = self.stage3(x, context_quarter)                     # [B, Base_C*2, H/4, W/4] → [B, Base_C*4, H/4, W/4]
        x = self.pool3(x3)                                       # [B, Base_C*4, H/4, W/4] → [B, Base_C*4, H/8, W/8]
        
        # Bridge with transformer attention
        x = self.bridge(x)                                       # [B, Base_C*4, H/8, W/8] → [B, Base_C*8, H/8, W/8]
        # Reshape for attention: [B, Base_C*8, H/8, W/8] → [B, H*W/64, Base_C*8]
        context_eighth = create_context_for_resolution(H//8, W//8)  # Context at eighth resolution
        x_bridge = x.view(B, self.base_channels * 8, (H//8) * (W//8)).permute(0, 2, 1)
        x_bridge = self.bridge_attn(x_bridge, context_eighth)    # Apply conditioned attention
        # Reshape back: [B, H*W/64, Base_C*8] → [B, Base_C*8, H/8, W/8]
        x = x_bridge.permute(0, 2, 1).view(B, self.base_channels * 8, H//8, W//8)
        
        # Decoder path - progressively increase spatial resolution, decrease channels
        x = self.up1(x)                                          # [B, Base_C*8, H/8, W/8] → [B, Base_C*8, H/4, W/4]
        x = torch.cat([x, x3], dim=1)                            # [B, Base_C*8+Base_C*4, H/4, W/4] = [B, Base_C*12, H/4, W/4]
        x = self.stage4(x, context_quarter)                      # [B, Base_C*12, H/4, W/4] → [B, Base_C*4, H/4, W/4]
        
        x = self.up2(x)                                          # [B, Base_C*4, H/4, W/4] → [B, Base_C*4, H/2, W/2]
        x = torch.cat([x, x2], dim=1)                            # [B, Base_C*4+Base_C*2, H/2, W/2] = [B, Base_C*6, H/2, W/2]
        x = self.stage5(x, context_half)                         # [B, Base_C*6, H/2, W/2] → [B, Base_C*2, H/2, W/2]
        
        x = self.up3(x)                                          # [B, Base_C*2, H/2, W/2] → [B, Base_C*2, H, W]
        x = torch.cat([x, x1], dim=1)                            # [B, Base_C*2+Base_C, H, W] = [B, Base_C*3, H, W]
        x = self.stage6(x, context_full)                         # [B, Base_C*3, H, W] → [B, Base_C, H, W]
        
        # Final output projection to spectral channels
        x = self.final(x)                                        # [B, Base_C, H, W] → [B, L, H, W]
        
        return x


if __name__ == "__main__":
    # Test the hyperspectral diffusion model
    batch_size = 2
    height, width = 64, 64
    spectral_channels = 128  # Number of hyperspectral bands
    sensor_channels = 16     # Number of sensor response channels
    
    model = U2NetHyperspectral(spectral_channels, sensor_channels)
    
    # Create test data
    x_t = torch.randn(batch_size, spectral_channels, height, width)
    context = torch.randn(batch_size, sensor_channels, height, width)
    t = torch.randint(0, 1000, (batch_size,))
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Input (x_t) shape: {x_t.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Time shape: {t.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x_t, t, context)
    
    print(f"Output shape: {output.shape}")
    print("Model test completed successfully!")
    
    # Test with different sizes to verify adaptability
    print("\nTesting different spatial resolutions:")
    for size in [(32, 32), (128, 128)]:
        h, w = size
        x_test = torch.randn(1, spectral_channels, h, w)
        context_test = torch.randn(1, sensor_channels, h, w)
        t_test = torch.randint(0, 1000, (1,))
        
        with torch.no_grad():
            out_test = model(x_test, t_test, context_test)
        
        print(f"Size {h}x{w}: Input {x_test.shape} → Output {out_test.shape}")
