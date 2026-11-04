"""HFD dataset that simulates the poor electrochromic sensor (Phase 3): per item it draws a measurement setting
(sequential / snapshot, SNR), measures the cube, and returns the noise-aware MMSE centre and posterior-std maps as
the conditioning. The cubes, split and file list are those of HFD_data; its y and R are never used."""
import hashlib
import os
import zlib

import numpy as np
import scipy.io as sio
import torch

from data_loader.HFD_dataset import HFD_data
from data_loader.linear_estimate import expand_matrix, load_prior
from data_loader.mmse import mmse_maps
from data_loader.sensor_config import training_sha
from data_loader.sensor_forward import measure, sigma_from_snr
from data_loader.sensor_model import load_sensor
from data_loader.warmstart_paths import PRIOR

MODES_FWD = ('sequential', 'snapshot')
MAP_KEYS = ('x', 'y', 'x0_hat', 's_post', 's_raw', 'mask')
SCALAR_KEYS = ('sigma', 'snr_db', 'snapshot')
SEED_SCHEME = 'crc32(relpath)'     # the setting and noise rngs are keyed by the cube's file, not its img_list position


def load_cube01(path):
    """The loader's per-cube normalisation of the .mat's `truth` onto [0, 1]; [H, W, 31] float64."""
    g = np.asarray(sio.loadmat(path)['truth'], dtype=np.float64)
    return (g - g.min()) / (g.max() - g.min() + 1e-20)


class HFDSensorSim(torch.utils.data.Dataset):
    def __init__(self, data_root, split, config, device='device1', setting=None, prior_path=PRIOR,
                 eval_ratio=0.1, type='Flower', R_override=None):
        need = ('seed', 's_floor01', 'snr_floor_db', 'P_y', 'family')
        missing = [k for k in need if k not in config]
        if not missing:
            missing = [k for k in ('snr_low_db', 'snr_high_db', 'p_floor', 'p_snapshot') if k not in config['family']]
        if missing:
            raise KeyError(f'sensor config is missing keys: {missing}')
        if setting is not None and setting['mode'] not in MODES_FWD:
            raise ValueError(f"unknown mode {setting['mode']!r}; expected one of {MODES_FWD}")
        self.data_root, self.prior_path, self.split, self.setting, self.config = \
            data_root, prior_path, split, setting, config
        base = HFD_data(data_path=data_root, train_mode='image', eval_ratio=eval_ratio, split=split,
                        data_format='image', type=type, R_n=1, sensor_down_sample_rate=1)
        self.img_list = base.img_list
        self.wavelens = base.wavelens
        self.sensor = load_sensor(device, data_root)
        self.R = self.sensor['R'] if R_override is None else R_override
        self.R_sha256 = self.sensor['sha256'] if R_override is None else \
            hashlib.sha256(np.ascontiguousarray(R_override, dtype=np.float64).tobytes()).hexdigest()
        p = load_prior(prior_path)
        self.prior = {k: p[k] for k in ('mu', 'C', 's') if k in p}
        self.A = expand_matrix(self.wavelens, 64)
        self.d = 1
        self.s_floor_tr = 2 * config['s_floor01']
        self.sigma_d = self.s_floor_tr
        self.P_y = config['P_y']
        self.epoch = 0
        self.collate_fn = sensor_collate_fn
        self.config_sha = training_sha(config)

    def __len__(self):
        return len(self.img_list)

    def set_epoch(self, e):
        self.epoch = int(e)

    def file_key(self, idx):
        """crc32 of the cube's path relative to data_root: the per-file key of the setting and noise rngs, so a cube
        gets the same measurement whatever its position in img_list (subsets pair by file)."""
        return zlib.crc32(os.path.relpath(self.img_list[idx], self.data_root).encode('utf-8'))

    def draw_setting(self, idx):
        if self.setting is not None:
            return self.setting['mode'], float(self.setting['snr_db'])
        c, f = self.config, self.config['family']
        rng = np.random.default_rng([c['seed'], self.epoch, self.file_key(idx)])
        mode = 'snapshot' if rng.random() < f['p_snapshot'] else 'sequential'
        if rng.random() < f['p_floor']:
            snr = float(c['snr_floor_db'])
        else:
            snr = float(rng.uniform(f['snr_low_db'], f['snr_high_db']))
        return mode, snr

    def __getitem__(self, idx):
        x01 = load_cube01(self.img_list[idx])
        mode, snr = self.draw_setting(idx)
        sigma = sigma_from_snr(snr, self.P_y)
        rng = np.random.default_rng([self.config['seed'] + 1, self.epoch, self.file_key(idx)])
        y, mask = measure(x01, self.R, mode, sigma, rng)
        snapshot = mode == 'snapshot'
        x0_hat, s64 = mmse_maps(y, mask if snapshot else None, self.R, self.prior, sigma, self.A)
        x = ((2.0 * x01 - 1.0) @ self.A.T).astype(np.float32)
        return dict(x=x, y=y, x0_hat=np.asarray(x0_hat, np.float32),
                    s_post=np.maximum(s64, self.s_floor_tr).astype(np.float32), s_raw=np.asarray(s64, np.float32),
                    mask=mask, sigma=np.float32(sigma), snr_db=np.float32(snr), snapshot=np.float32(snapshot))

    def checkpoint_extra(self):
        c = self.config
        return dict(sensor_config_sha=self.config_sha, s_floor01=c['s_floor01'], P_y=c['P_y'],
                    snr_floor_db=c['snr_floor_db'], family=c['family'], device_train=c.get('device_train'),
                    sensor_device=self.sensor['device'], R_sha256=self.R_sha256, seed_scheme=SEED_SCHEME)


def sensor_collate_fn(batch):
    out = {k: torch.from_numpy(np.stack([b[k] for b in batch], 0).astype(np.float32)).permute(0, 3, 1, 2).contiguous()
           for k in MAP_KEYS}
    for k in SCALAR_KEYS:
        out[k] = torch.tensor([float(b[k]) for b in batch], dtype=torch.float32)
    return out
