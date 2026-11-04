"""Adapter so DiffusionTrainer can keep calling model(x_t, cond, t) with a dict condition.

cond = {'sensor': [B, S, h, w] low-resolution sensor image, 'x0_hat': [B, L, H, W] warm-start estimate}
  use_x0_hat=True  (method B): net(cat([x_t, x0_hat], 1), sensor, t)   -- net needs extra_in_channels == L
  use_x0_hat=False (ablation A): net(x_t, sensor, t)
Sensor modes (Phase 3) take no context: use_context=False passes None (the U-Net then runs self-attention only), and
use_s_post=True appends cond['s_post'] [B, L, H, W] after x0_hat: net(cat([x_t, x0_hat, s_post], 1), None, t).
"""
import torch
import torch.nn as nn


class ResidualWrapper(nn.Module):
    def __init__(self, net: nn.Module, use_x0_hat: bool, use_s_post: bool = False, use_context: bool = True):
        super().__init__()
        self.net = net
        self.use_x0_hat = bool(use_x0_hat)
        self.use_s_post = bool(use_s_post)
        self.use_context = bool(use_context)

    def forward(self, x_t, cond, t):
        if not isinstance(cond, dict) or (self.use_context and 'sensor' not in cond):
            raise ValueError("cond must be a dict with keys 'sensor' and (for residual mode) 'x0_hat'")
        parts = [x_t]
        if self.use_x0_hat:
            parts.append(cond['x0_hat'])
        if self.use_s_post:
            parts.append(cond['s_post'])
        if len(parts) > 1:
            expected = getattr(self.net, 'input_proj').in_channels
            got = [p.shape[1] for p in parts]
            if sum(got) != expected:
                raise ValueError(f'net.input_proj expects {expected} channels, got {" + ".join(map(str, got))}')
            inp = torch.cat([p.to(x_t.dtype) for p in parts], dim=1)
        else:
            inp = x_t
        return self.net(inp, cond['sensor'] if self.use_context else None, t)


MODES = ('residual', 'standard', 'standard_xhat', 'regress', 'residual_reg', 'varhead',
         'uncond', 'cond_fam', 'regress_fam', 'varhead_fam')
# residual      (B):  input [x_t, x0_hat], target (x - x0_hat) / sigma_d
# standard      (A):  input  x_t,          target  x
# standard_xhat (A'): input [x_t, x0_hat], target  x   -- B's inputs with A's target: controls for the spatially
#                     aligned x0_hat channels that the position-blind cross-attention context cannot provide
# regress       (R):  input [0, x0_hat] at t = REGRESS_T, target (x - x0_hat) / sigma_d predicted in ONE deterministic
#                     call (no noise, no sampler): B's architecture in B's one-step regime, trained as a regressor
# residual_reg  (RR, Phase 2a): B with the regressor's cached estimate x_r in the x0_hat slot (data_loader.xr_cache):
#                     input [x_t, x_r], target (x - x_r) / sigma_r, v-prediction, sampled like B
# varhead       (VH, Phase 2a): input [x_r, x_r] at t = REGRESS_T, output log sigma^2 of (x - x_r) / sigma_r per
#                     element, Gaussian NLL, one deterministic call: the variance-head baseline around the same centre
# uncond        (U, Phase 3):  input  x_t,            no context, target  x   -- the unconditional prior
# cond_fam      (CF, Phase 3): input [x_t, x̂, s],     no context, target (x - x̂) / s per element: x̂ the MMSE centre,
#                     s the posterior-std map (the sufficient-statistic conditioning), sampled like B
# regress_fam   (RF, Phase 3): input [0, x̂, s] at t = REGRESS_T, no context, target (x - x̂) / s in ONE deterministic call
# varhead_fam   (VF, Phase 3): input [x̂, x̂, s] at t = REGRESS_T, no context, output log sigma^2 of (x - x̂) / s, Gaussian NLL
XR_MODES = ('residual_reg', 'varhead')                                  # the x0_hat slot holds x_r, ds.sigma_d is sigma_r
FAM_MODES = ('cond_fam', 'regress_fam', 'varhead_fam')                  # the x0_hat slot holds x̂, the s_post slot s
SENSOR_MODES = ('uncond',) + FAM_MODES                                  # no cross-attention context; dict batches
CENTRED_MODES = ('residual', 'regress', 'residual_reg', 'varhead') + FAM_MODES   # target (x - centre) / sigma
ONE_CALL_MODES = ('regress', 'varhead', 'regress_fam', 'varhead_fam')   # one deterministic call at t = REGRESS_T

REGRESS_T = 999     # the regressor's constant time input: B's one-step (ddim1) call is at t = 999, where x_t is pure noise
LOGVAR_RANGE = (-10.0, 10.0)      # varhead's log-variance is clamped here before the NLL (sigma between e^-5 and e^5)


def build_residual_model(mode, base_channels, spectral_channels=64, sensor_channels=30):
    """The one place that maps a mode name to an architecture; training and evaluation both call it."""
    from model.u2net_hyperspectral import U2NetHyperspectral      # local import: u2net_hyperspectral must not import this module
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
    use_x0_hat = mode in ('residual', 'standard_xhat', 'regress') + XR_MODES + FAM_MODES
    use_s_post = mode in FAM_MODES
    use_context = mode not in SENSOR_MODES
    extra = spectral_channels * (int(use_x0_hat) + int(use_s_post))
    net = U2NetHyperspectral(spectral_channels=spectral_channels, sensor_channels=sensor_channels,
                             base_channels=base_channels, extra_in_channels=extra)
    return ResidualWrapper(net, use_x0_hat=use_x0_hat, use_s_post=use_s_post, use_context=use_context)


def target_for(mode, x, x0_hat, scale):
    """The training target per mode (all on the loader's [-1, 1] scale); for XR_MODES x0_hat is x_r and scale is sigma_r,
    for FAM_MODES x0_hat is the MMSE centre and scale the posterior-std map (a tensor broadcastable to x)."""
    if mode in CENTRED_MODES:
        return (x - x0_hat) / scale
    return x


def batch_to_cond(mode, batch, sigma_d, device):
    """One loader batch -> (x, cond, scale) on `device`: a legacy tuple (x, y_d, xh) for the six legacy modes (scale is
    the scalar sigma_d), a dict batch (data_loader.hfd_sensor_sim) for SENSOR_MODES (scale is s_post, None for uncond)."""
    def mv(v):
        return v.to(device, non_blocking=True)
    if isinstance(batch, dict):
        if mode not in SENSOR_MODES:
            raise ValueError(f'dict (sensor) batches need a mode in {SENSOR_MODES}, got {mode!r}')
        if mode == 'uncond':
            return mv(batch['x']), {'sensor': None}, None
        s_post = mv(batch['s_post'])
        return mv(batch['x']), {'sensor': None, 'x0_hat': mv(batch['x0_hat']), 's_post': s_post}, s_post
    if mode in SENSOR_MODES:
        raise ValueError(f'mode {mode!r} needs dict (sensor) batches, got a {type(batch).__name__}')
    x, y_d, xh = batch
    return mv(x), {'sensor': mv(y_d), 'x0_hat': mv(xh)}, sigma_d


def regress_forward(model, cond):
    """The regressor's single call: a zero x_t placeholder (the 64 channels B reserves for x_t) at t = REGRESS_T.
    Training, reconstruction and the y-swap all call this, so the placeholder and t cannot drift apart."""
    xh = cond['x0_hat']
    t = torch.full((xh.shape[0],), REGRESS_T, dtype=torch.long, device=xh.device)
    return model(torch.zeros_like(xh), cond, t)


def varhead_forward(model, cond):
    """The variance head's single call: the centre x_r itself in the x_t slot (spec §4.5), at t = REGRESS_T.
    Returns the log-variance of the standardised residual, [B, 64, H, W], unclamped."""
    xr = cond['x0_hat']
    t = torch.full((xr.shape[0],), REGRESS_T, dtype=torch.long, device=xr.device)
    return model(xr, cond, t)


def varhead_nll(logvar, r):
    """Per-element Gaussian negative log-likelihood 0.5 (log s^2 + r^2 / s^2) of the standardised residual r under
    N(0, s^2), log s^2 clamped to LOGVAR_RANGE (the 0.5 log 2 pi constant is dropped)."""
    lv = logvar.clamp(*LOGVAR_RANGE)
    return 0.5 * (lv + r ** 2 * torch.exp(-lv))


def compose(centre, sigma, r):
    """x_hat = clamp(centre + sigma * r, -1, 1): one composition rule for the regressor, B and the Phase 2a samples."""
    return (centre + sigma * r).clamp(-1, 1)
