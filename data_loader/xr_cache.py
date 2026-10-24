"""The regressor-centre cache of Phase 2a (spec §4.3) and the dataset that reads it.

Layout: <cache_dir>/meta.json and <cache_dir>/<split>/<relpath without .mat>.npy, one float16 [H, W, 64] array per
cube: for split 'train' the estimate x_r of the half-data regressor that did NOT see the cube (cross-fitting), for
split 'val' the full selected regressor's estimate. meta.json names the three regressors (run, epoch, sha256,
train_half), sigma_r (RMS of x - x_r over the cross-fitted training cubes), sigma_full (the same on the monitoring
files with the full regressor) and the split's sha256.

XRCachedData wraps an HFDResidualData and returns (x64, y_d, x_r) with .sigma_d = sigma_r, so the trainer's loops see
B's interface with the regressor's estimate in the x0_hat slot (plan decision D1).
"""
import hashlib
import json
import os

import numpy as np
import torch

from data_loader.hfd_residual import residual_collate_fn

META_NAME = 'meta.json'
xr_collate_fn = residual_collate_fn


def cache_path(cache_dir, split, relpath):
    return os.path.join(cache_dir, split, os.path.splitext(relpath)[0] + '.npy')


def save_xr(cache_dir, split, relpath, x_r_hwc):
    path = cache_path(cache_dir, split, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp.npy'
    np.save(tmp, np.asarray(x_r_hwc, dtype=np.float16))
    os.replace(tmp, path)
    return path


def write_meta(cache_dir, meta):
    path = os.path.join(cache_dir, META_NAME)
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; a cache is built once (delete the directory to rebuild)')
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def read_meta(cache_dir):
    path = os.path.join(cache_dir, META_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found: not a centre cache (scripts/uq/build_xr_cache.py writes {META_NAME})')
    with open(path) as f:
        return json.load(f)


def meta_sha256(cache_dir):
    with open(os.path.join(cache_dir, META_NAME), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_crossfit_meta(meta):
    """The train half of each regressor must be what its role requires, or a cube was centred by a net that saw it."""
    want = dict(half1=1, half2=2, full=0)
    for key, th in want.items():
        got = (meta.get(key) or {}).get('train_half')
        if got != th:
            raise ValueError(f"cache meta {key}: train_half {got!r}, expected {th} (a '{key}' regressor must be trained "
                             f"on {'half ' + str(th) if th else 'the full training set'})")


class XRCachedData(torch.utils.data.Dataset):
    def __init__(self, base, cache_dir, split, data_root):
        self.base, self.cache_dir, self.split, self.data_root = base, cache_dir, split, data_root
        self.meta = read_meta(cache_dir)
        missing = [k for k in ('d', 'sigma_r', 'split_sha256', 'half1', 'half2', 'full') if k not in self.meta]
        if missing:                            # up front, not a KeyError here or in checkpoint_extra() after the model is built
            raise ValueError(f'cache {cache_dir} meta lacks {missing}')
        if int(self.meta['d']) != int(base.d):
            raise ValueError(f'cache {cache_dir} was built at d={int(self.meta["d"])} but the dataset is at d={int(base.d)}')
        check_crossfit_meta(self.meta)
        self.d = int(base.d)
        self.sigma_r = float(self.meta['sigma_r'])
        self.sigma_d = self.sigma_r            # the residual scale of THIS dataset's centre; the trainer reads ds.sigma_d
        self.prior = base.prior

    @property
    def img_list(self):
        return self.base.img_list

    @img_list.setter
    def img_list(self, value):
        self.base.img_list = value

    def path_of(self, file):
        return cache_path(self.cache_dir, self.split, os.path.relpath(file, self.data_root))

    def check_files(self):
        missing = [f for f in self.img_list if not os.path.isfile(self.path_of(f))]
        if missing:
            raise FileNotFoundError(f'{len(missing)} of {len(self.img_list)} cubes have no cached centre under '
                                    f'{self.cache_dir}/{self.split}, e.g. {os.path.relpath(missing[0], self.data_root)}')

    def checkpoint_extra(self):
        return dict(sigma_r=self.sigma_r, xr_cache_sha256=meta_sha256(self.cache_dir), centre=self.meta['full'],
                    split_sha256=self.meta['split_sha256'])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        x64, y_d = self.base.raw_item(idx)
        x_r = np.load(self.path_of(self.img_list[idx])).astype(np.float32)
        return x64, y_d, x_r
