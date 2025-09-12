import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *


class U2NetBlock1D(nn.Module):
    """
    A nested U-Net block for 1D data processing with transformer attention.
    
    This block implements a mini U-Net structure that processes data through multiple
    resolution scales while maintaining the sequence length relationship. The block
    uses a combination of convolutions for local feature extraction and transformer
    attention for global context.
    
    Dimension Changes Through the Network:
        Input: [B, Cin, L]           # B=batch, Cin=input channels, L=sequence length
        Stage1: [B, M, L]            # M=mid_channels
          ├─► Pool1: [B, M, L/2]     # Downsample sequence length
        Stage2: [B, M, L/2]
          ├─► Pool2: [B, M, L/4]     # Further downsample
        Stage3: [B, M, L/4]          # Bridge
          ├─► Up1: [B, M, L/2]       # Begin upsampling
        Stage4: [B, M, L/2]          # Concat with Stage2 features: [B, 2M, L/2] -> [B, M, L/2]
          ├─► Up2: [B, M, L]         # Final upsampling
        Stage5: [B, Cout, L]         # Cout=output channels, concatenated with Stage1
    
    Attention Operations:
        - Each stage includes a transformer block that processes data as [B, L, C]
        - Input is transposed from [B, C, L] to [B, L, C] for attention
        - After attention, data is transposed back to [B, C, L] for convolutions
    
    Skip Connections:
        Stage1 → Stage5: Preserves fine details from highest resolution
        Stage2 → Stage4: Preserves medium-scale features
    """
    def __init__(self, in_channels, mid_channels, out_channels, context_dim):
        """
        Initialize a U2NetBlock1D.

        Args:
            in_channels (int): Number of input channels (Cin)
            mid_channels (int): Number of channels in the intermediate layers (M)
                Used throughout the network for feature processing
            out_channels (int): Number of output channels (Cout)
                Used in the final stage to match desired output dimension
            context_dim (int): Dimension of the context vector from parent network
                Used for projecting global context to local features

        Channel Dimension Changes:
            1. Input         : [B, Cin, L]     → [B, M, L]      (stage1)
            2. Intermediate  : [B, M, L/2]     → [B, M, L/4]    (stage2,3)
            3. Skip Concat   : [B, 2M, L/2]    → [B, M, L/2]    (stage4)
            4. Final Concat  : [B, 2M, L]      → [B, Cout, L]   (stage5)

        Attention Dimensions:
            Each attention block processes:
            - Input : [B, C, L]   (conv format)
            - Trans : [B, L, C]   (attention format)
            - Output: [B, C, L]   (back to conv format)
            Where C is either mid_channels or out_channels
        """
        super(U2NetBlock1D, self).__init__()
        
        # Stage 1: Downsampling path [B, Cin, L] → [B, M, L] → [B, M, L/2]
        self.stage1 = ConvBlock1D(in_channels, mid_channels)              # [B, Cin, L] → [B, M, L]
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)               # [B, M, L] → [B, M, L/2]
        self.norm1 = nn.LayerNorm(mid_channels)                          # For attention: [B, L, M]
        # Context projection from global conditioning to local dimension
        self.context_proj1 = nn.Linear(context_dim, mid_channels)        # Project [B, L, context_dim] → [B, L, M]
        self.attn1 = BasicTransformerBlock(                              # [B, L, M] → [B, L, M]
            dim=mid_channels, n_heads=4, 
            d_head=mid_channels//4, gated_ff=True
        )
        
        # Stage 2: Downsampling path [B, M, L/2] → [B, M, L/2] → [B, M, L/4]
        self.stage2 = ConvBlock1D(mid_channels, mid_channels)            # [B, M, L/2] → [B, M, L/2]
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)               # [B, M, L/2] → [B, M, L/4]
        self.norm2 = nn.LayerNorm(mid_channels)                          # For attention: [B, L/2, M]
        self.context_proj2 = nn.Linear(context_dim, mid_channels)        # Project [B, L, context_dim] → [B, L, M]
        self.attn2 = BasicTransformerBlock(                              # [B, L/2, M] → [B, L/2, M]
            dim=mid_channels, n_heads=4, 
            d_head=mid_channels//4, gated_ff=True
        )
        
        # Stage 3: Bridge [B, M, L/4] → [B, M, L/4]
        self.stage3 = ConvBlock1D(mid_channels, mid_channels)            # [B, M, L/4] → [B, M, L/4]
        self.norm3 = nn.LayerNorm(mid_channels)                          # For attention: [B, L/4, M]
        self.context_proj3 = nn.Linear(context_dim, mid_channels)        # Project [B, L, context_dim] → [B, L, M]
        self.attn3 = BasicTransformerBlock(                              # [B, L/4, M] → [B, L/4, M]
            dim=mid_channels, n_heads=4, 
            d_head=mid_channels//4, gated_ff=True
        )
        
        # Stage 4: Upsampling path [B, M, L/4] → [B, 2M, L/2] → [B, M, L/2]
        self.up1 = nn.Upsample(scale_factor=2, mode='linear',            # [B, M, L/4] → [B, M, L/2]
                              align_corners=True)
        self.stage4 = ConvBlock1D(mid_channels * 2, mid_channels)        # [B, 2M, L/2] → [B, M, L/2]
        self.norm4 = nn.LayerNorm(mid_channels)                          # For attention: [B, L/2, M]
        self.context_proj4 = nn.Linear(context_dim, mid_channels)        # Project [B, L, context_dim] → [B, L, M]
        self.attn4 = BasicTransformerBlock(                              # [B, L/2, M] → [B, L/2, M]
            dim=mid_channels, n_heads=4, 
            d_head=mid_channels//4, gated_ff=True
        )
        
        # Stage 5: Upsampling path [B, M, L/2] → [B, 2M, L] → [B, Cout, L]
        self.up2 = nn.Upsample(scale_factor=2, mode='linear',            # [B, M, L/2] → [B, M, L]
                              align_corners=True)
        self.stage5 = ConvBlock1D(mid_channels * 2, out_channels)        # [B, 2M, L] → [B, Cout, L]
        self.norm5 = nn.LayerNorm(out_channels)                          # For attention: [B, L, Cout]
        self.context_proj5 = nn.Linear(context_dim, out_channels)        # Project [B, L, context_dim] → [B, L, Cout]
        self.attn5 = BasicTransformerBlock(                              # [B, L, Cout] → [B, L, Cout]
            dim=out_channels, n_heads=4, 
            d_head=out_channels//4, gated_ff=True
        )

    def forward(self, x, context=None):
        """
        Forward pass of the U2NetBlock1D.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, L]
                B = batch size
                C = number of channels
                L = sequence length
            context (torch.Tensor, optional): Context tensor for conditional attention
                Shape: [B, L, context_dim]

        Returns:
            torch.Tensor: Output tensor of shape [B, out_channels, L]

        Note:
            The block processes data through multiple scales with attention:
            1. Downsampling path: progressively reduces spatial dimensions
            2. Bridge: processes at lowest resolution
            3. Upsampling path: combines features from downsampling path
        """
        # Convert from [B, C, L] to [B, L, C] for attention
        def to_attention_format(x):
            return x.permute(0, 2, 1)
        
        def from_attention_format(x):
            return x.permute(0, 2, 1)
        
        # Encoder path with cross-attention
        # Stage 1: Downsampling path (L → L/2)
        x1 = self.stage1(x)                                   # [B, Cin, L]     → [B, M, L]
        x1 = to_attention_format(x1)                          # [B, M, L]       → [B, L, M]
        if context is not None:
            # Project context to match current dimension
            context_proj = self.context_proj1(context)        # [B, L, M*8]     → [B, L, M]
            x1 = self.attn1(x1, context_proj)                # [B, L, M]       → [B, L, M]
        else:
            x1 = self.attn1(x1, None)                        # Self-attention only
        x1 = from_attention_format(x1)                       # [B, L, M]       → [B, M, L]
        x = self.pool1(x1)                                   # [B, M, L]       → [B, M, L/2]
        
        # Stage 2: Downsampling path (L/2 → L/4)
        x2 = self.stage2(x)                                   # [B, M, L/2]     → [B, M, L/2]
        x2 = to_attention_format(x2)                          # [B, M, L/2]     → [B, L/2, M]
        if context is not None:
            context_proj = self.context_proj2(context)        # [B, L, M*8]     → [B, L, M]
            x2 = self.attn2(x2, context_proj)                # [B, L/2, M]     → [B, L/2, M]
        else:
            x2 = self.attn2(x2, None)                        # Self-attention only
        x2 = from_attention_format(x2)                       # [B, L/2, M]     → [B, M, L/2]
        x = self.pool2(x2)                                   # [B, M, L/2]     → [B, M, L/4]
        
        # Stage 3: Bridge (process at lowest resolution)
        x = self.stage3(x)                                   # [B, M, L/4]     → [B, M, L/4]
        x = to_attention_format(x)                           # [B, M, L/4]     → [B, L/4, M]
        if context is not None:
            context_proj = self.context_proj3(context)        # [B, L, M*8]     → [B, L, M]
            x = self.attn3(x, context_proj)                  # [B, L/4, M]     → [B, L/4, M]
        else:
            x = self.attn3(x, None)                          # Self-attention only
        x = from_attention_format(x)                         # [B, L/4, M]     → [B, M, L/4]
        
        # Stage 4: Upsampling path with skip connection from Stage 2
        x = self.up1(x)                                      # [B, M, L/4]     → [B, M, L/2]
        x = torch.cat([x, x2], dim=1)                        # [B, M, L/2]     → [B, 2M, L/2]
        x = self.stage4(x)                                   # [B, 2M, L/2]    → [B, M, L/2]
        x = to_attention_format(x)                           # [B, M, L/2]     → [B, L/2, M]
        if context is not None:
            context_proj = self.context_proj4(context)        # [B, L, M*8]     → [B, L, M]
            x = self.attn4(x, context_proj)                  # [B, L/2, M]     → [B, L/2, M]
        else:
            x = self.attn4(x, None)                          # Self-attention only
        x = from_attention_format(x)                         # [B, L/2, M]     → [B, M, L/2]
        
        # Stage 5: Upsampling path with skip connection from Stage 1
        x = self.up2(x)                                      # [B, M, L/2]     → [B, M, L]
        x = torch.cat([x, x1], dim=1)                        # [B, M, L]       → [B, 2M, L]
        x = self.stage5(x)                                   # [B, 2M, L]      → [B, Cout, L]
        x = to_attention_format(x)                           # [B, Cout, L]    → [B, L, Cout]
        if context is not None:
            context_proj = self.context_proj5(context)        # [B, L, M*8]     → [B, L, Cout]
            x = self.attn5(x, context_proj)                  # [B, L, Cout]    → [B, L, Cout]
        else:
            x = self.attn5(x, None)                  # Self-attention only
        x = from_attention_format(x)                # [B, L, Cout]    → [B, Cout, L]
        
        return x

class U2Net1D(nn.Module):
    """
    1D U-Net architecture with nested U-Net blocks and transformer-style attention,
    designed for conditional diffusion models similar to Stable Diffusion.
    
    Architecture Overview (B=batch, C=channels, L=sequence length, Base_C=base_channels):
        Input [B, C, L]
        ├── Stage1 (UNet Block) ─────────────────────────────────┐
        │   └── Pool1 [B, Base_C, L] → [B, Base_C, L/2]          │
        ├── Stage2 (UNet Block) ─────────────────────┐           │
        │   └── Pool2 [B, Base_C*2, L/2] → [B, Base_C*2, L/4]    │           
        ├── Stage3 (UNet Block) ──────────┐          │           │
        │   └── Pool3 [B, Base_C*4, L/4] → [B, Base_C*4, L/8]    │                   
        ├── Bridge [B, Base_C*8, L/8]     │          │           │
        │   └── Up1                       │          │           │
        ├── Stage4 (UNet Block) ←─────────┘          │           │
        │   └── Up2                                  │           │
        ├── Stage5 (UNet Block) ←────────────────────┘           │
        │   └── Up3                                              │
        └── Stage6 (UNet Block) ←────────────────────────────────┘
        
    Channel Dimensions (Base_C = base_channels):
        - Stage1: Base_C      (base_channels)
        - Stage2: Base_C*2    (doubles channels)
        - Stage3: Base_C*4    (doubles again)
        - Bridge: Base_C*8    (doubles at bridge)
        - Stage4: Base_C*4    (begins reducing)
        - Stage5: Base_C*2    (reduces further)
        - Stage6: Base_C      (back to base)
    
    Conditioning:
        - Time embedding: Converts timestep to B*4 dimensional vector
        - Condition projection: Maps condition to B*8 dimensional space
        - Both are combined and used in cross-attention throughout network
    """
    
    def __init__(self, input_channels, condition_dim, base_channels=64):
        """
        Args:
            input_channels (int): Number of input channels in the data
            condition_dim (int): Dimension of the condition vector for guidance
            base_channels (int): Base number of channels, controls network capacity
                               Each subsequent level multiplies this by 2
        """
        super(U2Net1D, self).__init__()
        
        # Time embedding [B] → [B, Base_C*8]
        # Maps diffusion timestep to match context dimension
        time_dim = base_channels * 8
        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),                                                   # [B] → [B, Base_C]
            nn.Linear(base_channels, time_dim),                                             # [B, Base_C] → [B, Base_C*8]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)                                                   # [B, Base_C*8] → [B, Base_C*8]
        )
        
        # Condition projection [B, cond_dim] → [B, Base_C*8]
        # Projects condition vector to match attention dimensions
        self.cond_proj = nn.Sequential(
            nn.Linear(condition_dim, base_channels * 8),                                    # [B, cond_dim] → [B, Base_C*8]
            nn.GELU(),
            nn.Linear(base_channels * 8, base_channels * 8)                                 # [B, Base_C*8] → [B, Base_C*8]
        )
        
        # Encoder Path: Progressive downsampling and channel expansion
        # Stage 1: Input level
        self.stage1 = U2NetBlock1D(input_channels, base_channels, base_channels, 
                                  context_dim=base_channels * 8)                            # [B, C, L] → [B, Base_C, L]
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)                                  # [B, Base_C, L] → [B, Base_C, L/2]
        
        # Stage 2: First downsampling level
        self.stage2 = U2NetBlock1D(base_channels, base_channels * 2, base_channels * 2,
                                  context_dim=base_channels * 8)                            # [B, Base_C, L/2] → [B, Base_C*2, L/2]
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)                                  # [B, Base_C*2, L/2] → [B, Base_C*2, L/4]
        
        # Stage 3: Second downsampling level
        self.stage3 = U2NetBlock1D(base_channels * 2, base_channels * 4, base_channels * 4,
                                  context_dim=base_channels * 8)                            # [B, Base_C*2, L/4] → [B, Base_C*4, L/4]
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)                                  # [B, B*4, L/4] → [B, B*4, L/8]
        
        # Bridge: Bottleneck with maximum channels and attention
        bridge_channels = base_channels * 8
        self.bridge = ConvBlock1D(base_channels * 4, bridge_channels)                       # [B, Base_C*4, L/8] → [B, Base_C*8, L/8]
        self.bridge_norm = nn.LayerNorm(bridge_channels)                                    # For attention: [B, L/8, Base_C*8]
        self.bridge_attn = BasicTransformerBlock(
            dim=bridge_channels,                                                            # Processes: [B, L/8, Base_C*8]
            n_heads=8,
            d_head=bridge_channels//8,
            gated_ff=True,
            dropout=0.0
        )
        
        # Decoder Path: Progressive upsampling and channel reduction
        # Stage 4: First upsampling level
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)           # [B, Base_C*8, L/8] → [B, Base_C*8, L/4]
        # Concat with stage3: [B, Base_C*8 + Base_C*4, L/4] = [B, Base_C*12, L/4]
        self.stage4 = U2NetBlock1D(base_channels * 12, base_channels * 4, base_channels * 4,
                                  context_dim=base_channels * 8)                            # [B, Base_C*12, L/4] → [B, Base_C*4, L/4]
        
        # Stage 5: Second upsampling level
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)           # [B, Base_C*4, L/4] → [B, Base_C*4, L/2]
        # Concat with stage2: [B, Base_C*4 + Base_C*2, L/2] = [B, Base_C*6, L/2]
        self.stage5 = U2NetBlock1D(base_channels * 6, base_channels * 2, base_channels * 2,
                                  context_dim=base_channels * 8)                            # [B, Base_C*6, L/2] → [B, Base_C*2, L/2]
        
        # Stage 6: Final upsampling level
        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)           # [B, Base_C*2, L/2] → [B, Base_C*2, L]
        # Concat with stage1: [B, Base_C*2 + Base_C, L] = [B, Base_C*3, L]
        self.stage6 = U2NetBlock1D(base_channels * 3, base_channels, base_channels,
                                  context_dim=base_channels * 8)                            # [B, Base_C*3, L] → [B, Base_C, L]
        
        # Final projection to match input channels
        self.final = nn.Conv1d(base_channels, input_channels, kernel_size=1)                # [B, B, L] → [B, C, L]

    def forward(self, x, t, cond):
        """
        Forward pass of U2Net1D.
        
        Args:
            x: Input tensor of shape [B, C, L] where:
               B = batch size
               C = number of input channels
               L = sequence length
            t: Time embedding tensor of shape [B]
            cond: Condition tensor of shape [B, condition_dim]
            
        Returns:
            Output tensor of shape [B, C, L] - same as input
            
        Note:
            The network follows a U-shaped architecture where:
            1. Encoder path progressively reduces sequence length (L→L/2→L/4→L/8) while 
               expanding channels (C→Base_C→Base_C*2→Base_C*4→Base_C*8)
            2. Bridge applies attention at the bottleneck (L/8, Base_C*8)
            3. Decoder path progressively increases sequence length (L/8→L/4→L/2→L) while
               reducing channels (Base_C*8→Base_C*4→Base_C*2→Base_C→C)
            4. Skip connections concatenate encoder features with decoder features at each level
                B = batch size
                C = number of channels
                L = sequence length
                Base_C = base channels
            t (torch.Tensor): Timestep values [B]
                Used for diffusion time embedding
            cond (torch.Tensor): Condition vector [B, condition_dim]
                Provides guidance for the diffusion process
        
        Returns:
            torch.Tensor: Output tensor [B, C, L]
                Has the same shape as input x
        
        Note:
            The network processes the input through multiple scales:
            1. Input [B, C, L]
            2. First level [B, Base_C, L]
            3. Second level [B, Base_C*2, L/2]
            4. Third level [B, Base_C*4, L/4]
            5. Bridge [B, Base_C*8, L/8]
            Then upsamples back to original resolution
        """
        # Time and condition embedding
        t_emb = self.time_embedding(t)                            # [B] → [B, Base_C*8]
        
        # Process condition and combine with time embedding
        context = self.cond_proj(cond)                            # [B, cond_dim] → [B, Base_C*8]
        context = context.view(*context.shape, -1)                # [B, Base_C*8] → [B, Base_C*8, 1]
        
        # Convert context to attention format
        context = context.permute(0, 2, 1)                        # [B, Base_C*8, 1] → [B, 1, Base_C*8]
        
        # Add time embedding to each position in sequence
        context = context + t_emb[:, None, :]                     # [B, 1, Base_C*8] + [B, 1, Base_C*8]
        context = context.expand(-1, x.shape[2], -1)              # [B, 1, Base_C*8] → [B, L, Base_C*8]
        
        # Encoder path - progressively reduce sequence length, increase channels
        x1 = self.stage1(x, context)                              # [B, C, L] → [B, Base_C, L]
        x = self.pool1(x1)                                        # [B, Base_C, L] → [B, Base_C, L/2]
        
        x2 = self.stage2(x, context)                              # [B, Base_C, L/2] → [B, Base_C*2, L/2]
        x = self.pool2(x2)                                        # [B, Base_C*2, L/2] → [B, Base_C*2, L/4]
        
        x3 = self.stage3(x, context)                              # [B, Base_C*2, L/4] → [B, Base_C*4, L/4]
        x = self.pool3(x3)                                        # [B, Base_C*4, L/4] → [B, Base_C*4, L/8]
        
        # Bridge with transformer attention
        x = self.bridge(x)                                        # [B, Base_C*4, L/8] → [B, Base_C*8, L/8]
        x = x.permute(0, 2, 1)                                    # [B, Base_C*8, L/8] → [B, L/8, Base_C*8]
        x = self.bridge_attn(x, context)                          # Apply conditioned attention
        x = x.permute(0, 2, 1)                                    # [B, L/8, Base_C*8] → [B, Base_C*8, L/8]
        
        # Decoder path - progressively increase sequence length, decrease channels
        x = self.up1(x)                                           # [B, Base_C*8, L/8] → [B, Base_C*8, L/4]
        x = torch.cat([x, x3], dim=1)                             # [B, Base_C*8+Base_C*4, L/4] = [B, Base_C*12, L/4]
        x = self.stage4(x, context)                               # [B, Base_C*12, L/4] → [B, Base_C*4, L/4]
        
        x = self.up2(x)                                           # [B, Base_C*4, L/4] → [B, Base_C*4, L/2]
        x = torch.cat([x, x2], dim=1)                             # [B, Base_C*4+Base_C*2, L/2] = [B, Base_C*6, L/2]
        x = self.stage5(x, context)                               # [B, Base_C*6, L/2] → [B, Base_C*2, L/2]
        
        x = self.up3(x)                                           # [B, Base_C*2, L/2] → [B, Base_C*2, L]
        x = torch.cat([x, x1], dim=1)                             # [B, Base_C*2+Base_C, L] = [B, Base_C*3, L]
        x = self.stage6(x, context)                               # [B, Base_C*3, L] → [B, Base_C, L]
        
        # Final output projection
        x = self.final(x)                                         # [B, Base_C, L] → [B, C, L]
        
        return x

if __name__ == "__main__":
    # Test the model
    batch_size = 4
    sequence_length = 256
    input_channels = 3
    condition_dim = 64
    
    model = U2Net1D(input_channels, condition_dim)
    x = torch.randn(batch_size, input_channels, sequence_length)
    cond = torch.randn(batch_size, condition_dim)
    t = torch.randint(0, 1000, (batch_size,))
    
    output = model(x, t, cond)
    print(f"Input shape: {x.shape}")
    print(f"Condition shape: {cond.shape}")
    print(f"Output shape: {output.shape}")
