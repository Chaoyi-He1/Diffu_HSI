import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class BasicTransformerBlock(nn.Module):
    """
    Basic transformer block as used in Stable Diffusion.
    Combines self-attention, cross-attention, and feed-forward layers.
    """
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True):
        super().__init__()
        # Self-attention
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        # Cross-attention
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        # Feed-forward with gating
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        """
        Forward pass through transformer block, following Stable Diffusion implementation
        
        Args:
            x (torch.Tensor): Input tensor [B, L, C]
                B = batch size
                L = sequence length
                C = channels (dim)
            context (torch.Tensor, optional): Cross-attention context [B, S, C]
                S = context sequence length
        
        Returns:
            torch.Tensor: Transformed tensor [B, L, C]
        """
        # Self attention with residual
        x = self.attn1(self.norm1(x)) + x
        # Cross attention with residual
        x = self.attn2(self.norm2(x), context=context) + x
        # Feed forward with residual
        x = self.ff(self.norm3(x)) + x
        return x

class CrossAttention(nn.Module):
    """
    Cross attention module as used in Stable Diffusion.
    Supports both self-attention (context=None) and cross-attention (context provided).
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.d_head = dim_head
        
        # Separate projections for query, key, value
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        # Output projection with dropout
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """
        Cross-attention forward pass
        
        Args:
            x: query tensor, shape [B, L, C]
                L = sequence length of queries 
                C = query_dim
            context: conditioning (used as K,V), shape [B, S, context_dim]
                S = sequence length of context
            mask: attention mask (optional)
        
        Returns:
            tensor of shape [B, L, C]
        """
        h = self.heads

        # If no context provided, use x for self-attention
        if context is None:
            context = x
        
        # Linear projections
        q = self.to_q(x)        # [B, L, C] -> [B, L, n_heads * d_head] (query from x)
        k = self.to_k(context)  # [B, S, context_dim] -> [B, S, n_heads * d_head] (key from context)
        v = self.to_v(context)  # [B, S, context_dim] -> [B, S, n_heads * d_head] (value from context)

        # Reshape and transpose for multi-head attention
        # [B, S/L, n_heads * d_head] -> [B, n_heads, S/L, d_head]
        q, k, v = map(lambda t: t.view(*t.shape[:-1], h, self.d_head).transpose(-2, -3), (q, k, v))

        # Attention scores
        # [B, n_heads, S, d_head] @ [B, n_heads, L, d_head] -> [B, n_heads, S, L]
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Apply mask if provided
        if exists(mask):
            mask = mask[:, None, :, None] * mask[:, None, None, :]
            sim.masked_fill_(~mask, -torch.finfo(sim.dtype).max)

        # Attention weights and value aggregation
        attn = sim.softmax(dim=-1)                     # [B, n_heads, S, L]
        # [B, n_heads, S, L] @ [B, n_heads, L, d_head] -> [B, n_heads, S, d_head]
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # Reshape back to original dimension
        # [B, n_heads, S, d_head] -> [B, S, n_heads * d_head]
        out = out.transpose(-2, -3).reshape(*context.shape[:-1], h * self.d_head)
        
        # Final projection
        # [B, S, n_heads * d_head] -> [B, S, C]
        return self.to_out(out)

class GEGLU(nn.Module):
    """
    Gated GLU variant as used in Stable Diffusion.
    Splits projection into two chunks and applies GELU activation to the gate.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    """
    Feed-forward network as used in Stable Diffusion.
    Uses GEGLU activation and supports GLU gating.
    """
    def __init__(self, dim, dim_out=None, mult=4, glu=True, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        
        # Use GEGLU if gating is enabled, otherwise standard GELU
        project_in = GEGLU(dim, inner_dim) if glu else nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )
    
    def forward(self, x):
        return self.net(x)

class ConvBlock1D(nn.Module):
    """
    Double convolution block with batch normalization and ReLU activation.
    Maintains the spatial/temporal dimension through proper padding.
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(ConvBlock1D, self).__init__()
        self.conv = nn.Sequential(
            # First conv layer
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),  # [B, in_ch, L] -> [B, out_ch, L]
            nn.BatchNorm1d(out_channels),  # Normalize each channel
            nn.ReLU(inplace=True),         # Non-linearity
            # Second conv layer
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1), # [B, out_ch, L] -> [B, out_ch, L]
            nn.BatchNorm1d(out_channels),  # Normalize each channel
            nn.ReLU(inplace=True)          # Non-linearity
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [B, C_in, L]
                B = batch size
                C_in = input channels
                L = sequence length
        
        Returns:
            torch.Tensor: Output tensor [B, C_out, L]
                C_out = output channels
                Sequence length L is preserved through padding
        """
        return self.conv(x)


class ConvBlock2D(nn.Module):
    """
    Double convolution block with group normalization and GELU activation.
    Maintains the spatial dimensions (H, W) through proper padding.
    Uses group normalization instead of batch norm for better stability in diffusion models.
    """
    def __init__(self, in_channels, out_channels, groups=8):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            groups (int): Number of groups for group normalization
        """
        super(ConvBlock2D, self).__init__()
        # Adjust groups to be compatible with channels
        groups = min(groups, min(in_channels, out_channels))
        
        self.conv = nn.Sequential(
            # First conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # [B, in_ch, H, W] -> [B, out_ch, H, W]
            nn.GroupNorm(groups, out_channels),  # Group normalization instead of batch norm
            nn.GELU(),                           # GELU activation for better gradient flow
            # Second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # [B, out_ch, H, W] -> [B, out_ch, H, W]
            nn.GroupNorm(groups, out_channels),  # Group normalization
            nn.GELU()                            # GELU activation
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [B, C_in, H, W]
                B = batch size
                C_in = input channels
                H = height
                W = width
        
        Returns:
            torch.Tensor: Output tensor [B, C_out, H, W]
                C_out = output channels
                Spatial dimensions H, W are preserved through padding
        """
        return self.conv(x)
    

class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding module similar to the one used in Stable Diffusion.
    Converts scalar timesteps into high-dimensional vectors using sinusoidal functions.
    """
    def __init__(self, dim):
        """
        Args:
            dim (int): Output dimension of the time embedding
                      Should be even as it uses sin/cos pairs
        """
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Create sinusoidal time embeddings
        
        Args:
            t (torch.Tensor): Timestep values [B]
                B = batch size
        
        Returns:
            torch.Tensor: Time embeddings [B, dim]
                Each timestep is encoded as a vector of dimension 'dim'
                using interleaved sine and cosine functions at different frequencies
        """
        half_dim = self.dim // 2
        # Scale factor for different frequency bands
        emb = math.log(10000) / half_dim
        # Create frequency bands
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)  # [half_dim]
        # Multiply timesteps by frequency bands
        emb = t[:, None] * emb[None, :]  # [B, half_dim]
        # Concatenate sin and cos embeddings
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # [B, dim]
        return emb
    