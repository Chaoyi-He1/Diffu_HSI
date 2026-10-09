"""HFD dataset that also returns the closed-form estimate x0_hat computed from the low-resolution sensor.

Same files, split, normalisation and strided sensor as HFD_data (which it subclasses); only the return
value changes to (x64, y_d, x0_hat).
"""
import numpy as np
import torch

from data_loader.HFD_dataset import HFD_data
from data_loader.linear_estimate import estimate_from_sensor, expand_matrix, load_prior


class HFDResidualData(HFD_data):
    def __init__(self, data_path, split, sensor_down_sample_rate, prior_path, eval_ratio=0.1, type='Flower',
                 require_sigma_d=True):
        """require_sigma_d=False is for scripts/compute_residual_scale.py only, which measures sigma_d; the
        dataset then has sigma_d = None. Everything else needs the prior's sigma_d for this d and fails without it."""
        super().__init__(data_path=data_path, train_mode='image', eval_ratio=eval_ratio, split=split,
                         data_format='image', type=type, R_n=1, sensor_down_sample_rate=sensor_down_sample_rate)
        self.d = int(sensor_down_sample_rate)
        self.prior = load_prior(prior_path)
        R_loader = np.asarray(self.sensor_R_matrix, dtype=np.float64)
        if self.prior['R'].shape != R_loader.shape or not np.allclose(self.prior['R'], R_loader):
            raise ValueError(f'prior {prior_path} was fitted for a different sensor matrix R than the loader uses '
                             f'(prior R {self.prior["R"].shape}, loader R {R_loader.shape}); '
                             f're-run scripts/fit_linear_prior.py and scripts/compute_residual_scale.py')
        self.A = expand_matrix(self.wavelens, 64)
        dv = [int(v) for v in self.prior['d_values']]
        if self.d in dv:
            self.sigma_d = float(self.prior['sigma_d'][dv.index(self.d)])
        elif require_sigma_d:
            raise ValueError(f"prior has no sigma_d for d={self.d}; "
                             f"run scripts/compute_residual_scale.py --d_values {self.d}")
        else:
            self.sigma_d = None

    def __getitem__(self, idx):
        x64, y_d = super().__getitem__(idx)                       # [H, W, 64], [h, w, 30]
        H, W = x64.shape[:2]
        # The sensor data is the float32 y_d the network consumes; x0_hat is a deterministic function of exactly
        # that array (the estimate itself runs in float64), so the baseline never sees more precision than the net.
        y_d = np.asarray(y_d, dtype=np.float32)
        x0_hat = estimate_from_sensor(y_d.astype(np.float64), self.prior, self.d, H, W, self.A)
        return np.asarray(x64, dtype=np.float32), y_d, np.asarray(x0_hat, dtype=np.float32)


def residual_collate_fn(batch):
    x, y, xh = zip(*batch)
    to = lambda a: torch.tensor(np.stack(a, 0), dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
    return to(x), to(y), to(xh)
