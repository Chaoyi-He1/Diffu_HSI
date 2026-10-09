"""Adapter so DiffusionTrainer can keep calling model(x_t, cond, t) with a dict condition.

cond = {'sensor': [B, S, h, w] low-resolution sensor image, 'x0_hat': [B, L, H, W] warm-start estimate}
  use_x0_hat=True  (method B): net(cat([x_t, x0_hat], 1), sensor, t)   -- net needs extra_in_channels == L
  use_x0_hat=False (ablation A): net(x_t, sensor, t)
"""
import torch
import torch.nn as nn


class ResidualWrapper(nn.Module):
    def __init__(self, net: nn.Module, use_x0_hat: bool):
        super().__init__()
        self.net = net
        self.use_x0_hat = bool(use_x0_hat)

    def forward(self, x_t, cond, t):
        if not isinstance(cond, dict) or 'sensor' not in cond:
            raise ValueError("cond must be a dict with keys 'sensor' and (for residual mode) 'x0_hat'")
        if self.use_x0_hat:
            x0_hat = cond['x0_hat']
            expected = getattr(self.net, 'input_proj').in_channels
            if x_t.shape[1] + x0_hat.shape[1] != expected:
                raise ValueError(f'net.input_proj expects {expected} channels, got {x_t.shape[1]} + {x0_hat.shape[1]}')
            inp = torch.cat([x_t, x0_hat.to(x_t.dtype)], dim=1)
        else:
            inp = x_t
        return self.net(inp, cond['sensor'], t)
