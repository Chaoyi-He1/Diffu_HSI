import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple, Optional, List, Union, Dict


class DiffusionTrainer:
    """A lightweight diffusion trainer with explicit hyperparameters.

    This trainer implements a DDPM-style forward noising process, a flexible
    `get_loss` that supports common prediction targets (epsilon / x0 / v),
    and a `sample` method implementing the standard ancestral DDPM sampler.

    The implementation intentionally keeps dependencies small so it can be
    adapted for unconditional or conditional models.
    """

    def __init__(
        self,
        *,
        n_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",  # currently only 'linear' supported
        device: Optional[torch.device] = None,
        loss_type: str = "l2",  # 'l1' or 'l2'
        prediction_type: str = "eps",  # 'eps' | 'x0' | 'v'
        lr: float = 2e-4,
        betas_adam: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        ema_decay: Optional[float] = 0.995,
        grad_clip: Optional[float] = None,
        clamp_x0: Optional[Tuple[float, float]] = (-1.0, 1.0),
        snr_gamma: Optional[float] = 5.0,
    ) -> None:
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.device = device
        self.n_timesteps = int(n_timesteps)
        self.loss_type = loss_type
        self.prediction_type = prediction_type
        self.lr = lr
        self.betas_adam = betas_adam
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.grad_clip = grad_clip
        self.clamp_x0 = clamp_x0
        # min-SNR-γ loss weighting (Hang et al. 2023). None or <=0 disables.
        self.snr_gamma = snr_gamma if (snr_gamma is not None and snr_gamma > 0) else None
        # number of model evaluations made by the most recent `sample` call
        self.last_nfe = 0

        # Build beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.n_timesteps, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown beta_schedule={beta_schedule}")

        # register constants as tensors on device
        self.betas = betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.one_minus_alphas = 1.0 - self.alphas

        # Precompute posterior variance (used for p(x_{t-1} | x_t, x0))
        # standard DDPM posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        alpha_bar_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alpha_bars[:-1]], dim=0)
        self.post_variance = self.betas * (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bars)

        if self.loss_type == "l1":
            print('L1 Loss used in diffusion trainer')
        else:
            print('MSE Loss used in diffusion trainer')
        if self.snr_gamma is not None:
            print(f'min-SNR-γ loss weighting enabled (γ={self.snr_gamma}, pred={self.prediction_type})')
            
    # ----- utilities: q(x_t | x_0) and helpers -----
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0).

        Returns x_t and the ground-truth Gaussian noise used (true_eps).
        """
        if noise is None:
            true_eps = torch.randn_like(x_0, device=self.device)
        else:
            true_eps = noise
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, *([1] * (x_0.dim() - 1)))
        sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, *([1] * (x_0.dim() - 1)))
        x_t = sqrt_ab * x_0 + sqrt_1_ab * true_eps
        return x_t, true_eps

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, pred_eps: torch.Tensor) -> torch.Tensor:
        """Given x_t and a predicted noise vector (pred_eps), recover an x_0 estimate.

        Args:
            x_t: noisy input at timestep t.
            t: long tensor of timesteps for each batch element.
            pred_eps: model-predicted epsilon (same shape as x_t).
        Returns:
            Reconstructed x_0 estimate.
        """
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, *([1] * (x_t.dim() - 1)))
        sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, *([1] * (x_t.dim() - 1)))
        return (x_t - sqrt_1_ab * pred_eps) / sqrt_ab

    def predict_eps_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor) -> torch.Tensor:
        """Given x_t and an x0 estimate, return the implied epsilon (pred_eps).

        Args:
            x_t: noisy input at timestep t.
            t: long tensor of timesteps for each batch element.
            x0_pred: model-predicted x_0 (same shape as x_t).
        Returns:
            pred_eps: predicted epsilon consistent with x_t and x0_pred.
        """
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, *([1] * (x_t.dim() - 1)))
        sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, *([1] * (x_t.dim() - 1)))
        return (x_t - sqrt_ab * x0_pred) / sqrt_1_ab

    # ----- loss API -----
    def _loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return nn.L1Loss()(pred, target)
        return nn.MSELoss()(pred, target)

    def get_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        *,
        loss_in_fp32: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute training loss for a batch.

        Supports models that accept signature model(x_t, cond, t) and predict according
        to `self.prediction_type`.

        Returns (loss, info) where info contains intermediate tensors (noise, x_t, prediction).
        """
        model = model.to(self.device)
        x_0 = x_0.to(self.device)
        batch = x_0.shape[0]

        if t is None:
            t = torch.randint(0, self.n_timesteps, (batch,), device=self.device, dtype=torch.long)
        # sample noisy inputs and obtain ground-truth noise
        x_t, true_eps = self.q_sample(x_0, t)

        # Model predicts depending on prediction_type
        model_out = model(x_t, cond, t)

        if self.prediction_type == "eps":
            # model_out is the predicted epsilon (pred_eps)
            target = true_eps
            pred = model_out
            x0_pred = self.predict_x0_from_eps(x_t, t, pred)
        elif self.prediction_type == "x0":
            target = x_0
            pred = model_out
            x0_pred = pred
        elif self.prediction_type == "v":
            # v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x0
            sqrt_ab = self.sqrt_alpha_bars[t].view(-1, *([1] * (x_t.dim() - 1)))
            sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, *([1] * (x_t.dim() - 1)))
            v = sqrt_ab * true_eps - sqrt_1_ab * x_0
            target = v
            pred = model_out
            # convert v prediction back to x0 for logging
            x0_pred = sqrt_ab * x_t - sqrt_1_ab * v
        else:
            raise ValueError(f"Unknown prediction_type={self.prediction_type}")

        if loss_in_fp32:
            pred_loss = pred.float()
            target_loss = target.float()
        else:
            pred_loss = pred
            target_loss = target

        # Per-sample reduction over all non-batch dims → shape [B]
        if self.loss_type == "l1":
            per_sample = (pred_loss - target_loss).abs()
        else:
            per_sample = (pred_loss - target_loss).pow(2)
        per_sample = per_sample.mean(dim=list(range(1, per_sample.dim())))

        if self.snr_gamma is not None:
            # SNR(t) = alpha_bar_t / (1 - alpha_bar_t); weight per Hang et al. 2023.
            snr = self.alpha_bars[t] / (1.0 - self.alpha_bars[t])
            snr_clamped = snr.clamp(max=self.snr_gamma)
            if self.prediction_type == "eps":
                w = snr_clamped / snr
            elif self.prediction_type == "v":
                w = snr_clamped / (snr + 1.0)
            else:  # x0
                w = snr_clamped
            w = w.to(per_sample.dtype).detach()
            loss = (w * per_sample).mean()
        else:
            loss = per_sample.mean()
        info = {
            "loss": loss.detach(),
            "x_t": x_t.detach(),
            "true_eps": true_eps.detach(),
            "x0_pred": x0_pred.detach(),
            "t": t.detach(),
        }

        # Add an explicit prediction key depending on prediction_type for clearer logging
        if self.prediction_type == "eps":
            info["pred_eps"] = pred.detach()
        elif self.prediction_type == "x0":
            info["pred_x0"] = pred.detach()
        elif self.prediction_type == "v":
            info["pred_v"] = pred.detach()
        return loss, info

    # ----- sampling -----
    def _model_to_x0_eps(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert raw model output to (x0_pred, eps_pred) for any prediction_type."""
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, *([1] * (x_t.dim() - 1)))
        sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, *([1] * (x_t.dim() - 1)))
        if self.prediction_type == "eps":
            pred_eps = model_out
            x0_pred = (x_t - sqrt_1_ab * pred_eps) / sqrt_ab
        elif self.prediction_type == "x0":
            x0_pred = model_out
            pred_eps = (x_t - sqrt_ab * x0_pred) / sqrt_1_ab
        elif self.prediction_type == "v":
            v = model_out
            x0_pred = sqrt_ab * x_t - sqrt_1_ab * v
            pred_eps = sqrt_ab * v + sqrt_1_ab * x_t
        else:
            raise ValueError(f"Unknown prediction_type={self.prediction_type}")
        return x0_pred, pred_eps

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        cond: Optional[torch.Tensor],
        shape: Tuple[int, ...],
        n_steps: Optional[int] = None,
        method: str = "ddpm",
        step_interval: int = 1,
        eta: float = 0.0,
        use_interpolated_alphas: bool = False,
        progress: bool = True,
        clamp_x0: Optional[Tuple[float, float]] = None,
        x_init: Optional[torch.Tensor] = None,
        t_start: Optional[int] = None,
    ) -> torch.Tensor:
        """Unified sampler. Choose method 'ddpm' (ancestral) or 'ddim'.

        Args:
            method: 'ddpm' or 'ddim'.
            step_interval: for 'ddpm', stride between timesteps (>=1).
            eta: for 'ddim', controls stochasticity (0 deterministic).
            use_interpolated_alphas: unused (kept for backward compatibility).
            clamp_x0: override instance default; if provided, clamps the x0 estimate
                      each step to this range (e.g. (-1, 1)). Pass `None` to use the
                      instance default (`self.clamp_x0`); pass `(-inf, inf)` or set
                      instance default to `None` to disable clamping.
            x_init: optional clean-space estimate of shape `shape`. With `t_start`, the chain
                    starts from sqrt(ab[t_start]) * x_init + sqrt(1 - ab[t_start]) * noise and
                    runs only the steps from t_start down to 0 (warm start). Both or neither.
            t_start: integer in [0, n_timesteps - 1]; 0 returns clamp(x_init) with no model call.
        Side effect: sets self.last_nfe to the number of model evaluations made.
        """
        method = method.lower()
        if method not in ("ddpm", "ddim"):
            raise ValueError("method must be 'ddpm' or 'ddim'")

        model = model.to(self.device)
        if n_steps is None:
            n_steps = self.n_timesteps

        clamp_range = clamp_x0 if clamp_x0 is not None else self.clamp_x0

        def _maybe_clamp(x0_pred: torch.Tensor) -> torch.Tensor:
            if clamp_range is None:
                return x0_pred
            return x0_pred.clamp(clamp_range[0], clamp_range[1])

        if (x_init is None) != (t_start is None):
            raise ValueError('x_init and t_start must be given together')
        warm = x_init is not None
        if warm:
            if not (0 <= int(t_start) < self.n_timesteps):
                raise ValueError(f't_start must be in [0, {self.n_timesteps - 1}]')
            t_start = int(t_start)
            x_init = x_init.to(self.device)
            if t_start == 0:
                self.last_nfe = 0
                return _maybe_clamp(x_init)
            ab = self.alpha_bars[t_start]
            x = torch.sqrt(ab) * x_init + torch.sqrt(1.0 - ab) * torch.randn(shape, device=self.device)
            n_eff = min(n_steps, t_start + 1)
            grid = torch.linspace(t_start, 0, n_eff + 1).round().long().tolist()[:-1]   # descending, excludes 0
            ts = sorted(set(grid), reverse=True)
        else:
            x = torch.randn(shape, device=self.device)
            # Build descending timestep schedule (respaced if n_steps < n_timesteps)
            step_interval = max(1, self.n_timesteps // n_steps)
            ts = list(range(self.n_timesteps - 1, -1, -step_interval))
        self.last_nfe = len(ts)
        iterator = tqdm(ts, desc=f"{method.upper()} Sampling") if progress else ts

        B = shape[0]

        for i, timestep in enumerate(iterator):
            t = torch.full((B,), timestep, device=self.device, dtype=torch.long)
            # prev timestep (the one we transition TO); -1 means "output x_0"
            prev_timestep = ts[i + 1] if i + 1 < len(ts) else -1

            model_out = model(x, cond, t)
            x0_pred, _ = self._model_to_x0_eps(model_out, x, t)

            # (a) clamp x0 estimate — prevents 1/sqrt(alpha_bar_t) ≈ 143× error
            #     amplification at high t from compounding across sampling steps.
            x0_pred = _maybe_clamp(x0_pred)

            alpha_bar_t = self.alpha_bars[t].view(-1, *([1] * (x.dim() - 1)))

            if prev_timestep < 0:
                # Final step: return the clamped x0 estimate directly.
                x = x0_pred
                continue

            t_prev = torch.full((B,), prev_timestep, device=self.device, dtype=torch.long)
            alpha_bar_prev = self.alpha_bars[t_prev].view(-1, *([1] * (x.dim() - 1)))

            if method == "ddpm":
                # Strided ancestral update in x0 parameterization.
                # Let alpha_tilde = alpha_bar_t / alpha_bar_prev (product of alphas
                # between prev and t — reduces to alpha_t when step_interval == 1).
                alpha_tilde = alpha_bar_t / alpha_bar_prev
                beta_tilde = 1.0 - alpha_tilde
                one_minus_ab_t = 1.0 - alpha_bar_t
                # Posterior mean q(x_prev | x_t, x0):
                #   mu = (sqrt(ab_prev) * beta_tilde / (1 - ab_t)) * x0
                #      + (sqrt(alpha_tilde) * (1 - ab_prev) / (1 - ab_t)) * x_t
                coef_x0 = torch.sqrt(alpha_bar_prev) * beta_tilde / one_minus_ab_t
                coef_xt = torch.sqrt(alpha_tilde) * (1.0 - alpha_bar_prev) / one_minus_ab_t
                mean = coef_x0 * x0_pred + coef_xt * x
                # Posterior variance
                var = beta_tilde * (1.0 - alpha_bar_prev) / one_minus_ab_t
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var.clamp(min=0.0)) * noise
            else:  # DDIM
                # Recompute eps consistent with the (clamped) x0 so clamping
                # actually influences the next iterate.
                sqrt_1_ab_t = torch.sqrt(1.0 - alpha_bar_t)
                eps_consistent = (x - torch.sqrt(alpha_bar_t) * x0_pred) / sqrt_1_ab_t
                sigma = eta * torch.sqrt(
                    (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                    * (1.0 - alpha_bar_t / alpha_bar_prev)
                )
                dir_xt = torch.sqrt((1.0 - alpha_bar_prev - sigma ** 2).clamp(min=0.0)) * eps_consistent
                x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
                if eta > 0:
                    x = x + sigma * torch.randn_like(x)

        return x


