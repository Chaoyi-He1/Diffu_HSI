"""HFD dataset that also returns the closed-form estimate x0_hat computed from the low-resolution sensor.

Same files, split, normalisation and strided sensor as HFD_data (which it subclasses); only the return
value changes to (x64, y_d, x0_hat).
"""
import numpy as np
import torch

from data_loader.HFD_dataset import HFD_data
from data_loader.linear_estimate import estimate_from_sensor, expand_matrix, load_prior


class HFDResidualData(HFD_data):
    def __init__(self, data_path, split, sensor_down_sample_rate, prior_path, eval_ratio=0.1, type='Flower'):
        super().__init__(data_path=data_path, train_mode='image', eval_ratio=eval_ratio, split=split,
                         data_format='image', type=type, R_n=1, sensor_down_sample_rate=sensor_down_sample_rate)
        self.d = int(sensor_down_sample_rate)
        self.prior = load_prior(prior_path)
        self.A = expand_matrix(self.wavelens, 64)
        dv = list(self.prior['d_values'].tolist()) if self.prior['d_values'].size else []
        self.sigma_d = float(self.prior['sigma_d'][dv.index(self.d)]) if self.d in dv else 1.0

    def __getitem__(self, idx):
        x64, y_d = super().__getitem__(idx)                       # [H, W, 64], [h, w, 30]
        H, W = x64.shape[:2]
        x0_hat = estimate_from_sensor(np.asarray(y_d, dtype=np.float64), self.prior, self.d, H, W, self.A)
        return (np.asarray(x64, dtype=np.float32), np.asarray(y_d, dtype=np.float32),
                np.asarray(x0_hat, dtype=np.float32))


def residual_collate_fn(batch):
    x, y, xh = zip(*batch)
    to = lambda a: torch.tensor(np.stack(a, 0), dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
    return to(x), to(y), to(xh)
