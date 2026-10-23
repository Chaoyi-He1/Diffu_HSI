"""Phase 2a Task 3: the centre cache stores x_r per cube as float16 next to a meta that names the regressors, sigma_r
and the split; XRCachedData substitutes x_r for x0_hat and sigma_r for sigma_d and refuses a cache of another stride,
a cache whose halves are crossed, and missing files."""
import hashlib
import json
import os

import numpy as np
import pytest
import torch

from data_loader.hfd_residual import residual_collate_fn
from data_loader.xr_cache import (META_NAME, XRCachedData, cache_path, check_crossfit_meta, meta_sha256, read_meta,
                                  save_xr, write_meta, xr_collate_fn)

ROOT = '/fake/data root'


class StubBase:
    """What XRCachedData needs from HFDResidualData: d, prior, img_list and raw_item(idx) -> (x64, y_d) HWC."""
    def __init__(self, n=4, H=8, d=4):
        g = torch.Generator().manual_seed(0)
        self.d, self.prior = d, {'s': 3e-7}
        self.img_list = [os.path.join(ROOT, f'MatFlower60/Train/P00{1 + i % 2}/{i}_1.mat') for i in range(n)]
        self.x = (torch.rand(n, H, H, 64, generator=g) * 2 - 1).numpy()
        self.y = torch.rand(n, H // d, H // d, 30, generator=g).numpy()

    def raw_item(self, idx):
        return self.x[idx], self.y[idx]


def meta_for(d=4, **over):
    m = dict(d=d, split_sha256='abc', sigma_r=0.09, sigma_full=0.095, sigma_ratio=1.0556, dtype='float16',
             half1=dict(run='regress_l1_half1_d4_b64', epoch=12, sha256='h1', train_half=1),
             half2=dict(run='regress_l1_half2_d4_b64', epoch=14, sha256='h2', train_half=2),
             full=dict(run='regress_l1_sel_d4_b64', epoch=17, sha256='f', train_half=0))
    m.update(over)
    return m


def build_cache(tmp_path, base, meta, split='train', skip=()):
    cd = str(tmp_path / 'xr_cache')
    write_meta(cd, meta)
    xr = {}
    for i, f in enumerate(base.img_list):
        rel = os.path.relpath(f, ROOT)
        xr[f] = np.clip(base.x[i] + 0.09 * np.random.default_rng(i).standard_normal(base.x[i].shape), -1, 1).astype(np.float32)
        if rel not in skip:
            save_xr(cd, split, rel, xr[f])
    return cd, xr


def test_cache_path_layout():
    assert cache_path('/c', 'train', 'MatFlower60/Train/P001/3_2.mat') == '/c/train/MatFlower60/Train/P001/3_2.npy'


def test_meta_round_trip_and_sha(tmp_path):
    cd = str(tmp_path / 'c')
    write_meta(cd, meta_for())
    assert read_meta(cd) == meta_for()
    raw = open(os.path.join(cd, META_NAME), 'rb').read()
    assert meta_sha256(cd) == hashlib.sha256(raw).hexdigest()
    with pytest.raises(FileExistsError):
        write_meta(cd, meta_for())


def test_crossfit_meta_check():
    check_crossfit_meta(meta_for())
    with pytest.raises(ValueError, match='half1'):
        check_crossfit_meta(meta_for(half1=dict(run='r', epoch=1, sha256='x', train_half=2)))
    with pytest.raises(ValueError, match='full'):
        check_crossfit_meta(meta_for(full=dict(run='r', epoch=1, sha256='x', train_half=1)))


def test_items_substitute_the_cached_centre_and_sigma_r(tmp_path):
    base = StubBase()
    cd, xr = build_cache(tmp_path, base, meta_for())
    ds = XRCachedData(base, cd, 'train', ROOT)
    ds.check_files()
    assert len(ds) == 4 and ds.d == 4 and ds.sigma_d == 0.09 and ds.sigma_r == 0.09 and ds.prior is base.prior
    x, y, r = ds[1]
    assert x.dtype == y.dtype == r.dtype == np.float32
    assert np.array_equal(x, base.x[1]) and np.array_equal(y, base.y[1])
    assert np.abs(r - xr[base.img_list[1]]).max() <= 1e-3           # float16 storage of a [-1, 1] value
    bx, by, br = xr_collate_fn([ds[0], ds[1]])
    assert xr_collate_fn is residual_collate_fn and bx.shape == (2, 64, 8, 8) and by.shape == (2, 30, 2, 2) and br.shape == (2, 64, 8, 8)


def test_img_list_is_shared_with_the_base_so_filters_apply_to_both(tmp_path):
    base = StubBase()
    cd, _ = build_cache(tmp_path, base, meta_for())
    ds = XRCachedData(base, cd, 'train', ROOT)
    ds.img_list = ds.img_list[:2]
    assert len(ds) == 2 and base.img_list == ds.img_list
    x, _, _ = ds[1]
    assert np.array_equal(x, base.x[1])


def test_checkpoint_extra_names_sigma_r_cache_sha_centre_and_split(tmp_path):
    base = StubBase()
    cd, _ = build_cache(tmp_path, base, meta_for())
    ds = XRCachedData(base, cd, 'train', ROOT)
    e = ds.checkpoint_extra()
    assert e == dict(sigma_r=0.09, xr_cache_sha256=meta_sha256(cd), centre=meta_for()['full'], split_sha256='abc')


def test_a_cache_of_another_stride_is_refused(tmp_path):
    base = StubBase(d=8)
    cd, _ = build_cache(tmp_path, base, meta_for(d=4))
    with pytest.raises(ValueError, match='d=8'):
        XRCachedData(base, cd, 'train', ROOT)


def test_a_crossed_cache_is_refused(tmp_path):
    base = StubBase()
    cd, _ = build_cache(tmp_path, base, meta_for(half2=dict(run='r', epoch=1, sha256='x', train_half=1)))
    with pytest.raises(ValueError, match='half2'):
        XRCachedData(base, cd, 'train', ROOT)


def test_missing_files_are_reported_by_check_files(tmp_path):
    base = StubBase()
    cd, _ = build_cache(tmp_path, base, meta_for(), skip=('MatFlower60/Train/P002/1_1.mat',))
    ds = XRCachedData(base, cd, 'train', ROOT)
    with pytest.raises(FileNotFoundError, match='1_1'):
        ds.check_files()
    ds.img_list = [f for f in ds.img_list if '1_1' not in f]
    ds.check_files()


def test_a_missing_meta_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match=META_NAME):
        read_meta(str(tmp_path / 'nothing'))
