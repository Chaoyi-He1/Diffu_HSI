import os
import numpy as np
import pytest
import torch

PRIOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'results', 'residual_warmstart', 'linear_prior_R1.npz')
needs_prior = pytest.mark.skipif(not os.path.exists(PRIOR), reason='run scripts/fit_linear_prior.py first')


@pytest.mark.dataset
@needs_prior
def test_item_shapes_and_consistency(data_root):
    from data_loader.hfd_residual import HFDResidualData
    from data_loader.HFD_dataset import HFD_data
    ds = HFDResidualData(data_path=data_root, split='test', sensor_down_sample_rate=4, prior_path=PRIOR)
    base = HFD_data(data_path=data_root, train_mode='image', eval_ratio=0.1, split='test',
                    data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=4)
    assert len(ds) == len(base) == 1000
    x, y, xh = ds[0]
    bx, by = base[0]
    assert x.shape == (64, 64, 64) and y.shape == (16, 16, 30) and xh.shape == (64, 64, 64)
    assert np.allclose(x, bx, atol=1e-6) and np.allclose(y, by, atol=1e-6)   # same x and y as HFD_data
    assert xh.min() >= -1.0 and xh.max() <= 1.0
    assert np.sqrt(((xh - x) ** 2).mean()) < 0.2                              # a sensible estimate, not garbage


@pytest.mark.dataset
@needs_prior
def test_collate(data_root):
    from data_loader.hfd_residual import HFDResidualData, residual_collate_fn
    ds = HFDResidualData(data_path=data_root, split='test', sensor_down_sample_rate=2, prior_path=PRIOR)
    x, y, xh = residual_collate_fn([ds[0], ds[1]])
    assert x.shape == (2, 64, 64, 64) and y.shape == (2, 30, 32, 32) and xh.shape == (2, 64, 64, 64)
    assert x.dtype == torch.float32 and xh.dtype == torch.float32


@pytest.mark.dataset
@needs_prior
def test_sigma_d_is_unit_rms_after_scale_script(data_root):
    from data_loader.hfd_residual import HFDResidualData
    from data_loader.linear_estimate import load_prior
    prior = load_prior(PRIOR)
    if prior['sigma_d'].size == 0:
        pytest.skip('run scripts/compute_residual_scale.py first')
    ds = HFDResidualData(data_path=data_root, split='train', sensor_down_sample_rate=4, prior_path=PRIOR)
    rng = np.random.default_rng(0)
    r2 = []
    for i in rng.choice(len(ds), 40, replace=False):
        x, _, xh = ds[int(i)]
        r2.append(((x - xh) / ds.sigma_d) ** 2)
    rms = np.sqrt(np.mean(r2))
    assert 0.8 < rms < 1.2
