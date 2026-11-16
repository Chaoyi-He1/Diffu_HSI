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
from data_loader.mmse import mmse_maps, mmse_maps_general
from data_loader.scenarios import has_naive, has_pixels, measure_scenario, needs_consts, scenario_R, state_indices
from data_loader.sensor_config import training_sha
from data_loader.sensor_forward import measure, sigma_from_snr
from data_loader.sensor_model import load_sensor
from data_loader.warmstart_paths import PRIOR

MODES_FWD = ('sequential', 'snapshot')
MAP_KEYS = ('x', 'y', 'x0_hat', 's_post', 's_raw', 'mask')
SCALAR_KEYS = ('sigma', 'snr_db', 'snapshot')
# Phase 3b scenario items add these maps (noise_var always; the *_naive ones for poisson / saturation / quantisation;
# measured, the [H, W, 1] pixel flag, with a pixel pattern); sensor_collate_fn carries the ones the items hold
EXTRA_MAP_KEYS = ('noise_var', 'y_naive', 'mask_naive', 'noise_var_naive', 'x0_hat_naive', 's_post_naive', 's_raw_naive',
                  'measured')
SEED_SCHEME = 'crc32(relpath)'     # the setting and noise rngs are keyed by the cube's file, not its img_list position


def load_cube01(path):
    """The loader's per-cube normalisation of the .mat's `truth` onto [0, 1]; [H, W, 31] float64."""
    g = np.asarray(sio.loadmat(path)['truth'], dtype=np.float64)
    return (g - g.min()) / (g.max() - g.min() + 1e-20)


class HFDSensorSim(torch.utils.data.Dataset):
    def __init__(self, data_root, split, config, device='device1', setting=None, prior_path=PRIOR,
                 eval_ratio=0.1, type='Flower', R_override=None, scenario=None, scenario_consts=None):
        need = ('seed', 's_floor01', 'snr_floor_db', 'P_y', 'family')
        missing = [k for k in need if k not in config]
        if not missing:
            missing = [k for k in ('snr_low_db', 'snr_high_db', 'p_floor', 'p_snapshot') if k not in config['family']]
        if missing:
            raise KeyError(f'sensor config is missing keys: {missing}')
        if setting is not None and setting['mode'] not in MODES_FWD:
            raise ValueError(f"unknown mode {setting['mode']!r}; expected one of {MODES_FWD}")
        if scenario is not None:
            if setting is None:
                raise ValueError('a scenario needs a fixed setting (mode sequential, snr_db)')
            if setting['mode'] != 'sequential':
                raise ValueError(f"a scenario is measured in sequential mode, got mode {setting['mode']!r}")
            if device != 'device1':
                raise ValueError(f'a scenario is built on device1 (blend reads device2 itself), got device {device!r}')
            if R_override is not None:
                raise ValueError('a scenario defines its own R; R_override is not allowed with it')
            if needs_consts(scenario) and scenario_consts is None:
                raise ValueError(f"noise kind {scenario['noise']['kind']!r} needs the scenario constants (scenario_consts)")
        self.data_root, self.prior_path, self.split, self.setting, self.config = \
            data_root, prior_path, split, setting, config
        self.scenario, self.scenario_consts = scenario, scenario_consts
        base = HFD_data(data_path=data_root, train_mode='image', eval_ratio=eval_ratio, split=split,
                        data_format='image', type=type, R_n=1, sensor_down_sample_rate=1)
        self.img_list = base.img_list
        self.wavelens = base.wavelens
        self.sensor = load_sensor(device, data_root)
        self.R = self.sensor['R'] if R_override is None else R_override
        self.R_sha256 = self.sensor['sha256'] if R_override is None else \
            hashlib.sha256(np.ascontiguousarray(R_override, dtype=np.float64).tobytes()).hexdigest()
        self.state_indices = None                # Phase 3b: the columns of the 30-state R a scenario keeps
        if scenario is not None:
            if scenario.get('states') is not None:
                self.state_indices = state_indices(self.sensor['R'].shape[1], scenario['states'])
            self.R = scenario_R(self.sensor['R'], load_sensor('device2', data_root)['R'], scenario)
            self.R_sha256 = hashlib.sha256(np.ascontiguousarray(self.R, dtype=np.float64).tobytes()).hexdigest()
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
        if self.scenario is not None:
            return self._scenario_item(idx)
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

    def _scenario_item(self, idx):
        """The scenario's aware measurement and statistics (mmse_maps_general), plus the naive ones and the pixel flag
        when the scenario has them. The rng is the Phase 3 noise rng (same per-file key); measure_scenario draws the
        pixel pattern from it first, then the noise."""
        x01 = load_cube01(self.img_list[idx])
        mode, snr = self.draw_setting(idx)
        sigma = sigma_from_snr(snr, self.P_y)
        rng = np.random.default_rng([self.config['seed'] + 1, self.epoch, self.file_key(idx)])
        m = measure_scenario(x01, self.R, sigma, self.scenario, rng, self.scenario_consts)
        x0_hat, s64 = mmse_maps_general(m['y'], m['mask'], m['noise_var'], self.R, self.prior, self.A)
        x = ((2.0 * x01 - 1.0) @ self.A.T).astype(np.float32)
        out = dict(x=x, y=m['y'], x0_hat=np.asarray(x0_hat, np.float32),
                   s_post=np.maximum(s64, self.s_floor_tr).astype(np.float32), s_raw=np.asarray(s64, np.float32),
                   mask=m['mask'], sigma=np.float32(sigma), snr_db=np.float32(snr), snapshot=np.float32(0.0),
                   noise_var=m['noise_var'])
        if has_naive(self.scenario):
            xn, sn = mmse_maps_general(m['y_naive'], m['mask_naive'], m['noise_var_naive'], self.R, self.prior, self.A)
            out.update(y_naive=m['y_naive'], mask_naive=m['mask_naive'], noise_var_naive=m['noise_var_naive'],
                       x0_hat_naive=np.asarray(xn, np.float32),
                       s_post_naive=np.maximum(sn, self.s_floor_tr).astype(np.float32), s_raw_naive=np.asarray(sn, np.float32))
        if has_pixels(self.scenario):
            out['measured'] = m['measured'][..., None].astype(np.float32)
        return out

    def checkpoint_extra(self):
        c = self.config
        return dict(sensor_config_sha=self.config_sha, s_floor01=c['s_floor01'], P_y=c['P_y'],
                    snr_floor_db=c['snr_floor_db'], family=c['family'], device_train=c.get('device_train'),
                    sensor_device=self.sensor['device'], R_sha256=self.R_sha256, seed_scheme=SEED_SCHEME)


def sensor_collate_fn(batch):
    out = {k: torch.from_numpy(np.stack([b[k] for b in batch], 0).astype(np.float32)).permute(0, 3, 1, 2).contiguous()
           for k in MAP_KEYS}
    for k in EXTRA_MAP_KEYS:                   # scenario items only; absent keys leave the Phase 3 batch unchanged
        if k in batch[0]:
            out[k] = torch.from_numpy(np.stack([b[k] for b in batch], 0).astype(np.float32)).permute(0, 3, 1, 2).contiguous()
    for k in SCALAR_KEYS:
        out[k] = torch.tensor([float(b[k]) for b in batch], dtype=torch.float32)
    return out
