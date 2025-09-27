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

    def get_loss(self, model: nn.Module, x_0: torch.Tensor, cond: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        loss = self._loss_fn(pred, target)
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
    ) -> torch.Tensor:
        """Unified sampler. Choose method 'ddpm' (ancestral) or 'ddim'.

        Args:
            method: 'ddpm' or 'ddim'.
            step_interval: for 'ddpm', stride between timesteps (>=1).
            eta: for 'ddim', controls stochasticity (0 deterministic).
            use_interpolated_alphas: for 'ddim', whether to interpolate alpha_bars.
        """
        method = method.lower()
        if method not in ("ddpm", "ddim"):
            raise ValueError("method must be 'ddpm' or 'ddim'")

        model = model.to(self.device)
        if n_steps is None:
            n_steps = self.n_timesteps

        x = torch.randn(shape, device=self.device)

        if method == "ddpm":
            if step_interval < 1:
                raise ValueError("step_interval must be >= 1")

            iterator = range(n_steps - 1, -1, -step_interval)
            if progress:
                iterator = tqdm(iterator, desc="DDPM Sampling")

            for timestep in iterator:
                # Generate noise for all timesteps except the last one
                z = torch.randn_like(x) if timestep > 0 else torch.zeros_like(x)
                
                # Get timestep tensor for batch
                t = torch.full((shape[0],), timestep, device=self.device, dtype=torch.long)
                
                # Get model prediction
                model_out = model(x, cond, t)

                # Convert model output to epsilon prediction
                if self.prediction_type == "eps":
                    pred_eps = model_out
                elif self.prediction_type == "x0":
                    x0_pred = model_out
                    pred_eps = self.predict_eps_from_x0(x, t, x0_pred)
                elif self.prediction_type == "v":
                    sqrt_ab = self.sqrt_alpha_bars[t].view(-1, *([1] * (x.dim() - 1)))
                    sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, *([1] * (x.dim() - 1)))
                    v = model_out
                    # Convert v-prediction to x0: x0 = sqrt(alpha_bar) * x_t - sqrt(1-alpha_bar) * v
                    x0_pred = sqrt_ab * x - sqrt_1_ab * v
                    pred_eps = self.predict_eps_from_x0(x, t, x0_pred)
                else:
                    raise ValueError(f"Unknown prediction_type={self.prediction_type}")

                # Get schedule parameters
                alpha = self.alphas[t].view(-1, *([1] * (x.dim() - 1)))
                alpha_bar = self.alpha_bars[t].view(-1, *([1] * (x.dim() - 1)))
                
                # DDPM sampling step: x_{t-1} = c0 * (x_t - c1 * eps) + sigma * z
                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                
                # Apply the DDPM update
                x = c0 * (x - c1 * pred_eps) + torch.sqrt(self.post_variance[t].view(-1, *([1] * (x.dim() - 1)))) * z

            return x

        # DDIM path - based on diffusion_util.py reference
        iterator = range(n_steps - 1, -1, -step_interval)
        if progress:
            iterator = tqdm(iterator, desc="DDIM Sampling")

        for timestep in iterator:
            # Get timestep tensor for batch
            t = torch.full((shape[0],), timestep, device=self.device, dtype=torch.long)
            
            # Get model prediction
            model_out = model(x, cond, t)

            # Convert model output to epsilon prediction
            if self.prediction_type == "eps":
                pred_eps = model_out
            elif self.prediction_type == "x0":
                x0_pred = model_out
                pred_eps = self.predict_eps_from_x0(x, t, x0_pred)
            elif self.prediction_type == "v":
                sqrt_ab = self.sqrt_alpha_bars[t].view(-1, *([1] * (x.dim() - 1)))
                sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, *([1] * (x.dim() - 1)))
                v = model_out
                # Convert v-prediction to x0: x0 = sqrt(alpha_bar) * x_t - sqrt(1-alpha_bar) * v
                x0_pred = sqrt_ab * x - sqrt_1_ab * v
                pred_eps = self.predict_eps_from_x0(x, t, x0_pred)
            else:
                raise ValueError(f"Unknown prediction_type={self.prediction_type}")

            # Get schedule parameters
            alpha_bar = self.alpha_bars[t].view(-1, *([1] * (x.dim() - 1)))
            
            # Calculate next timestep alpha_bar
            next_timestep = max(0, timestep - step_interval)
            alpha_bar_next = self.alpha_bars[next_timestep].view(-1, *([1] * (x.dim() - 1)))
            
            # DDIM sampling step: x0_t = (x_t - eps * sqrt(1-alpha_bar)) / sqrt(alpha_bar)
            #                     x_next = sqrt(alpha_bar_next) * x0_t + sqrt(1-alpha_bar_next) * eps
            x0_t = (x - pred_eps * torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_bar)
            x = torch.sqrt(alpha_bar_next) * x0_t + torch.sqrt(1 - alpha_bar_next) * pred_eps

        return x


