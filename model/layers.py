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
        # [B, L/S, n_heads * d_head] -> [B, n_heads, L/S, d_head]
        q, k, v = map(lambda t: t.view(*t.shape[:-1], h, self.d_head).transpose(-2, -3), (q, k, v))

        # Prefer SDPA (Flash / mem-efficient attention on CUDA): avoids materializing
        # full [B, h, L, S] similarity — critical for full-res U-Net stages (large L).
        if not exists(mask):
            # q: [B, h, Lq, d], k,v: [B, h, Lk, d] -> out: [B, h, Lq, d]
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            # Explicit attention (O(L*S) memory) when a mask is required
            sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            mask = mask[:, None, :, None] * mask[:, None, None, :]
            sim.masked_fill_(~mask, -torch.finfo(sim.dtype).max)
            attn = sim.softmax(dim=-1)
            out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # [B, h, Lq, d] -> [B, Lq, h*d]
        out = out.transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[1], h * self.d_head)
        
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

    When `time_dim` is provided, applies FiLM (feature-wise linear modulation)
    after each GroupNorm — each conv layer gets its own per-channel
    (scale, shift) predicted from t_emb. FiLM linears are zero-initialized so
    the block starts as identity and backward compatibility is preserved when
    `forward` is called without t_emb (e.g. under nn.Sequential).
    """
    def __init__(self, in_channels, out_channels, time_dim=None, groups=8):
        super(ConvBlock2D, self).__init__()
        groups = min(groups, min(in_channels, out_channels))

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

        if time_dim is not None:
            self.film1 = nn.Linear(time_dim, 2 * out_channels)
            self.film2 = nn.Linear(time_dim, 2 * out_channels)
            for lin in (self.film1, self.film2):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        else:
            self.film1 = None
            self.film2 = None

    def forward(self, x, t_emb=None):
        h = self.norm1(self.conv1(x))
        if self.film1 is not None and t_emb is not None:
            scale, shift = self.film1(t_emb).chunk(2, dim=-1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        if self.film2 is not None and t_emb is not None:
            scale, shift = self.film2(t_emb).chunk(2, dim=-1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.act(h)
    

class ChannelSpatialConv(nn.Module):
    """
    Convolution that slides over the channel, height, and width dimensions simultaneously.

    Internally reshapes [B, C, H, W] → [B, 1, C, H, W] and applies `num_filters`
    parallel Conv3d filters over the (C, H, W) volume. The filter axis is then
    reduced back to 1 with a 1×1×1 Conv3d, and the leading dim is squeezed so
    the output is [B, C, H, W] — channel count is preserved.

    `num_filters=1` recovers the original single-filter behaviour. Using >1
    multiplies the block's expressivity without changing its I/O signature.

    Args:
        spatial_kernel (int): Kernel size for H and W dimensions (default 3).
        channel_kernel (int): Kernel size for the channel dimension (default 3).
        num_filters (int): Number of parallel 3D filters applied inside the
            block, mixed down to 1 by a pointwise 1×1×1 Conv3d (default 4).
    """
    def __init__(
        self,
        spatial_kernel: int = 3,
        channel_kernel: int = 3,
        num_filters: int = 4,
    ):
        super().__init__()
        if num_filters < 1:
            raise ValueError(f"num_filters must be >= 1, got {num_filters}")
        self.num_filters = num_filters
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(channel_kernel, spatial_kernel, spatial_kernel),
            padding=(channel_kernel // 2, spatial_kernel // 2, spatial_kernel // 2),
        )
        self.mix = nn.Conv3d(num_filters, 1, kernel_size=1) if num_filters > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, 1, C, H, W]
        y = self.conv3d(x.unsqueeze(1))  # [B, K, C, H, W]
        if self.mix is not None:
            y = self.mix(y)              # [B, 1, C, H, W]
        return y.squeeze(1)              # [B, C, H, W]


class ConvBlock2D_ChannelAware(nn.Module):
    """
    Double conv block that convolves over channel, height, and width simultaneously.

    Unlike ConvBlock2D (which only convolves over H and W), this block treats the
    channel axis as a spatial dimension so the kernel can capture cross-channel
    locality in addition to spatial locality.

    Channel projection logic:
      - out_channels > in_channels : up-project channels first (1×1 Conv2d),
                                     then apply channel-spatial conv at out_channels.
      - out_channels < in_channels : apply channel-spatial conv at in_channels,
                                     then down-project channels (1×1 Conv2d).
      - out_channels == in_channels: apply channel-spatial conv directly, no projection.

    Args:
        in_channels (int): Input channel count.
        out_channels (int): Output channel count.
        spatial_kernel (int): Kernel size for H and W dimensions (default 3).
        channel_kernel (int): Kernel size for the channel dimension (default 3).
        num_filters (int): Number of parallel 3D filters inside each
            ChannelSpatialConv (default 4). Higher ⇒ more spectral expressivity.
        groups (int): Number of groups for GroupNorm (default 8).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_kernel: int = 3,
        channel_kernel: int = 3,
        num_filters: int = 4,
        time_dim: int = None,
        groups: int = 8,
    ):
        super().__init__()

        # --- channel projection ---
        self.up_proj = None
        self.down_proj = None

        if out_channels > in_channels:
            self.up_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            work_ch = out_channels
        elif out_channels < in_channels:
            self.down_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            work_ch = in_channels   # conv runs at in_channels, projection happens after
        else:
            work_ch = in_channels

        g = min(groups, work_ch)

        # --- double channel-spatial conv (split so FiLM can inject between GN and GELU) ---
        self.cs_conv1 = ChannelSpatialConv(spatial_kernel, channel_kernel, num_filters)
        self.norm1 = nn.GroupNorm(g, work_ch)
        self.cs_conv2 = ChannelSpatialConv(spatial_kernel, channel_kernel, num_filters)
        self.norm2 = nn.GroupNorm(g, work_ch)
        self.act = nn.GELU()

        if time_dim is not None:
            self.film1 = nn.Linear(time_dim, 2 * work_ch)
            self.film2 = nn.Linear(time_dim, 2 * work_ch)
            for lin in (self.film1, self.film2):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        else:
            self.film1 = None
            self.film2 = None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]
            t_emb: optional [B, time_dim] time embedding; ignored if FiLM
                layers were not constructed (time_dim=None).
        Returns:
            [B, out_channels, H, W]
        """
        if self.up_proj is not None:
            x = self.up_proj(x)          # [B, in_ch→out_ch, H, W]

        h = self.norm1(self.cs_conv1(x))
        if self.film1 is not None and t_emb is not None:
            scale, shift = self.film1(t_emb).chunk(2, dim=-1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.act(h)
        h = self.norm2(self.cs_conv2(h))
        if self.film2 is not None and t_emb is not None:
            scale, shift = self.film2(t_emb).chunk(2, dim=-1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.act(h)

        if self.down_proj is not None:
            h = self.down_proj(h)        # [B, work_ch→out_ch, H, W]
        return h


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
    