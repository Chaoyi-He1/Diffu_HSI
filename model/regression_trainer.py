"""Objective of the one-step residual regressor (mode 'regress'), duck-typed to DiffusionTrainer.get_loss so the
training script's loops (train_one_epoch, validate, dry_run) run it unchanged."""
from model.residual_wrapper import REGRESS_T, regress_forward


class RegressionTrainer:
    prediction_type = 'direct'

    def __init__(self, *, device, loss_type='l2', n_timesteps=1000):
        if loss_type not in ('l1', 'l2'):
            raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type!r}")
        self.device = device
        self.loss_type = loss_type
        self.n_timesteps = int(n_timesteps)       # provenance only: the regressor has no noise schedule
        self.last_nfe = 1

    def get_loss(self, model, x_0, cond=None, t=None, *, loss_in_fp32=False):
        """x_0 is the regression target r = (x - x0_hat) / sigma_d; `t` is ignored (the call is at REGRESS_T).
        Returns (mean over samples of the per-sample mean |r_hat - r| or (r_hat - r)^2, info)."""
        x_0 = x_0.to(self.device)
        pred = regress_forward(model, cond)
        p, r = (pred.float(), x_0.float()) if loss_in_fp32 else (pred, x_0)
        err = (p - r).abs() if self.loss_type == 'l1' else (p - r) ** 2
        loss = err.reshape(err.shape[0], -1).mean(dim=1).mean()
        return loss, dict(loss=loss.detach(), pred=pred.detach(), t=REGRESS_T)
