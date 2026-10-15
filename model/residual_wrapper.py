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


MODES = ('residual', 'standard', 'standard_xhat', 'regress')
# residual      (B):  input [x_t, x0_hat], target (x - x0_hat) / sigma_d
# standard      (A):  input  x_t,          target  x
# standard_xhat (A'): input [x_t, x0_hat], target  x   -- B's inputs with A's target: controls for the spatially
#                     aligned x0_hat channels that the position-blind cross-attention context cannot provide
# regress       (R):  input [0, x0_hat] at t = REGRESS_T, target (x - x0_hat) / sigma_d predicted in ONE deterministic
#                     call (no noise, no sampler): B's architecture in B's one-step regime, trained as a regressor

REGRESS_T = 999     # the regressor's constant time input: B's one-step (ddim1) call is at t = 999, where x_t is pure noise


def build_residual_model(mode, base_channels, spectral_channels=64, sensor_channels=30):
    """The one place that maps a mode name to an architecture; training and evaluation both call it."""
    from model.u2net_hyperspectral import U2NetHyperspectral      # local import: u2net_hyperspectral must not import this module
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
    use_x0_hat = mode in ('residual', 'standard_xhat', 'regress')
    net = U2NetHyperspectral(spectral_channels=spectral_channels, sensor_channels=sensor_channels,
                             base_channels=base_channels, extra_in_channels=spectral_channels if use_x0_hat else 0)
    return ResidualWrapper(net, use_x0_hat=use_x0_hat)


def target_for(mode, x, x0_hat, sigma_d):
    """The training target per mode (all on the loader's [-1, 1] scale)."""
    if mode in ('residual', 'regress'):
        return (x - x0_hat) / sigma_d
    return x


def regress_forward(model, cond):
    """The regressor's single call: a zero x_t placeholder (the 64 channels B reserves for x_t) at t = REGRESS_T.
    Training, reconstruction and the y-swap all call this, so the placeholder and t cannot drift apart."""
    xh = cond['x0_hat']
    t = torch.full((xh.shape[0],), REGRESS_T, dtype=torch.long, device=xh.device)
    return model(torch.zeros_like(xh), cond, t)
