"""Objective of the variance-head baseline (mode 'varhead', spec §4.5), duck-typed to DiffusionTrainer.get_loss like
model.regression_trainer.RegressionTrainer so the training script's loops run it unchanged."""
from model.residual_wrapper import REGRESS_T, varhead_forward, varhead_nll


class VarheadTrainer:
    prediction_type = 'direct'
    loss_type = 'nll'

    def __init__(self, *, device, n_timesteps=1000):
        self.device = device
        self.n_timesteps = int(n_timesteps)       # provenance only
        self.last_nfe = 1

    def get_loss(self, model, x_0, cond=None, t=None, *, loss_in_fp32=False):
        """x_0 is the standardised residual r = (x - x_r) / sigma_r; the mean is fixed at x_r, the network predicts
        log sigma^2 per element. Returns (mean over samples of the per-sample mean NLL, info)."""
        x_0 = x_0.to(self.device)
        logvar = varhead_forward(model, cond)
        lv, r = (logvar.float(), x_0.float()) if loss_in_fp32 else (logvar, x_0)
        nll = varhead_nll(lv, r)
        loss = nll.reshape(nll.shape[0], -1).mean(dim=1).mean()
        return loss, dict(loss=loss.detach(), logvar=logvar.detach(), t=REGRESS_T)
