"""Sensor super-resolution U²-Net for diffusion.

Mirrors U2NetHyperspectral but:
  - in_channels == out_channels == sensor_channels (default 30)
  - context_encoder consumes a low-resolution sensor tensor
    [B, sensor_channels, H/ds, W/ds] and produces features that are
    resampled inside forward() to each stage's spatial resolution.

The U²-Net backbone (U2NetBlock2D), time embedding, attention, and
use_channel_3d_conv toggle are reused verbatim from the existing
implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    BasicTransformerBlock,
    ConvBlock2D,
    ConvBlock2D_ChannelAware,
    TimeEmbedding,
)
from .u2net_hyperspectral import U2NetBlock2D


class U2NetSensorSR(nn.Module):
    """Diffusion denoiser for sensor super-resolution.

    Args:
        sensor_channels (int): channel count for both x_t and cond (default 30)
        base_channels (int): network width
        use_channel_3d_conv (bool): switch backbone conv blocks to 3D
        channel_kernel, spatial_kernel, channel_num_filters: forwarded
            to ConvBlock2D_ChannelAware when use_channel_3d_conv=True.
    """

    def __init__(
        self,
        sensor_channels=30,
        base_channels=64,
        use_channel_3d_conv=False,
        channel_kernel=3,
        spatial_kernel=3,
        channel_num_filters=4,
    ):
        super().__init__()
        self.sensor_channels = sensor_channels
        self.base_channels = base_channels
        time_dim = base_channels * 8

        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Context encoder: consumes low-res sensor, produces base_C*8 features
        # at low-res. Spatial alignment to each stage is done in forward().
        self.context_encoder = nn.Sequential(
            nn.Conv2d(sensor_channels, base_channels,
                      kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2,
                      kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 8,
                      kernel_size=1),
        )

        self.input_proj = nn.Conv2d(
            sensor_channels, base_channels, kernel_size=3, padding=1
        )

        block_kwargs = dict(
            time_dim=time_dim,
            use_channel_3d_conv=use_channel_3d_conv,
            channel_kernel=channel_kernel,
            spatial_kernel=spatial_kernel,
            channel_num_filters=channel_num_filters,
        )

        # Bridge conv selection (mirrors u2net_hyperspectral.py:356-364)
        if use_channel_3d_conv:
            BridgeConvCls = ConvBlock2D_ChannelAware
            bridge_conv_kwargs = dict(
                channel_kernel=channel_kernel,
                spatial_kernel=spatial_kernel,
                num_filters=channel_num_filters,
                time_dim=time_dim,
            )
        else:
            BridgeConvCls = ConvBlock2D
            bridge_conv_kwargs = dict(time_dim=time_dim)

        # --- encoder ---
        self.stage1 = U2NetBlock2D(
            base_channels, base_channels, base_channels,
            context_dim=base_channels * 8, **block_kwargs,
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.stage2 = U2NetBlock2D(
            base_channels, base_channels * 2, base_channels * 2,
            context_dim=base_channels * 8, **block_kwargs,
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.stage3 = U2NetBlock2D(
            base_channels * 2, base_channels * 4, base_channels * 4,
            context_dim=base_channels * 8, **block_kwargs,
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # --- bridge ---
        bridge_channels = base_channels * 8
        self.bridge = BridgeConvCls(
            base_channels * 4, bridge_channels, **bridge_conv_kwargs
        )
        self.bridge_norm = nn.LayerNorm(bridge_channels)
        self.bridge_attn = BasicTransformerBlock(
            dim=bridge_channels,
            n_heads=8,
            d_head=bridge_channels // 8,
            gated_ff=True,
            dropout=0.0,
        )

        # --- decoder ---
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)
        self.stage4 = U2NetBlock2D(
            base_channels * 12, base_channels * 4, base_channels * 4,
            context_dim=base_channels * 8, **block_kwargs,
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)
        self.stage5 = U2NetBlock2D(
            base_channels * 6, base_channels * 2, base_channels * 2,
            context_dim=base_channels * 8, **block_kwargs,
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)
        self.stage6 = U2NetBlock2D(
            base_channels * 3, base_channels, base_channels,
            context_dim=base_channels * 8, **block_kwargs,
        )

        self.final = nn.Conv2d(
            base_channels, sensor_channels, kernel_size=1
        )

    def forward(self, x_t, cond, t):
        """
        Args:
            x_t  : [B, sensor_channels, H, W]        — noisy full-res sensor
            cond : [B, sensor_channels, H/ds, W/ds]  — low-res conditioning
            t    : [B]                                — timestep
        Returns:
            [B, sensor_channels, H, W]
        """
        B, _, H, W = x_t.shape

        t_emb = self.time_embedding(t)  # [B, base_C*8]

        # Encode low-res context → [B, base_C*8, Hc, Wc]
        context_feat = self.context_encoder(cond)

        def context_for(h, w):
            # Bilinear resample to (h, w), flatten to [B, h*w, base_C*8],
            # add time embedding broadcast across the spatial axis.
            ctx = F.interpolate(
                context_feat, size=(h, w),
                mode='bilinear', align_corners=True,
            )
            ctx = ctx.view(B, self.base_channels * 8, h * w).permute(0, 2, 1)
            return ctx + t_emb[:, None, :].expand(-1, h * w, -1)

        x = self.input_proj(x_t)  # [B, base_C, H, W]

        ctx_full = context_for(H, W)
        x1 = self.stage1(x, ctx_full, t_emb)
        x = self.pool1(x1)

        ctx_half = context_for(H // 2, W // 2)
        x2 = self.stage2(x, ctx_half, t_emb)
        x = self.pool2(x2)

        ctx_quarter = context_for(H // 4, W // 4)
        x3 = self.stage3(x, ctx_quarter, t_emb)
        x = self.pool3(x3)

        # Bridge
        x = self.bridge(x, t_emb)
        ctx_eighth = context_for(H // 8, W // 8)
        xb = x.view(B, self.base_channels * 8, (H // 8) * (W // 8)) \
              .permute(0, 2, 1)
        xb = self.bridge_attn(xb, ctx_eighth)
        x = xb.permute(0, 2, 1).view(
            B, self.base_channels * 8, H // 8, W // 8
        )

        # Decoder
        x = self.up1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.stage4(x, ctx_quarter, t_emb)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.stage5(x, ctx_half, t_emb)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.stage6(x, ctx_full, t_emb)

        return self.final(x)


if __name__ == '__main__':
    B, H, W = 2, 64, 64
    S = 30
    ds = 8

    for tag, flag in [('2dconv', False), ('3dconv', True)]:
        model = U2NetSensorSR(
            sensor_channels=S,
            base_channels=32,
            use_channel_3d_conv=flag,
            channel_kernel=7,
            spatial_kernel=3,
            channel_num_filters=4,
        )
        x_t = torch.randn(B, S, H, W)
        cond = torch.randn(B, S, H // ds, W // ds)
        t = torch.randint(0, 1000, (B,))
        with torch.no_grad():
            y = model(x_t, cond, t)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'[{tag}] params={n:,} out={y.shape}')
        assert y.shape == x_t.shape
    print('OK')
