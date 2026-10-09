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
@pytest.mark.parametrize('d', [1, 4])
def test_x0_hat_is_a_function_of_the_returned_float32_y_d(data_root, d):
    """The sensor data is the float32 y_d the network consumes; x0_hat must be computed from exactly that."""
    from data_loader.hfd_residual import HFDResidualData
    from data_loader.linear_estimate import estimate_from_sensor
    ds = HFDResidualData(data_path=data_root, split='test', sensor_down_sample_rate=d, prior_path=PRIOR)
    for i in (0, 500):
        x, y, xh = ds[i]
        assert y.dtype == np.float32 and xh.dtype == np.float32
        again = estimate_from_sensor(y, ds.prior, d, x.shape[0], x.shape[1], ds.A).astype(np.float32)
        assert np.array_equal(again, xh)


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
@pytest.mark.parametrize('d', [1, 2, 4, 8])
def test_sigma_d_is_unit_rms_after_scale_script(data_root, d):
    from data_loader.hfd_residual import HFDResidualData
    from data_loader.linear_estimate import load_prior
    prior = load_prior(PRIOR)
    if d not in [int(v) for v in prior['d_values']]:
        pytest.skip(f'run scripts/compute_residual_scale.py --d_values {d} first')
    ds = HFDResidualData(data_path=data_root, split='train', sensor_down_sample_rate=d, prior_path=PRIOR)
    rng = np.random.default_rng(0)
    r2 = []
    for i in rng.choice(len(ds), 40, replace=False):
        x, _, xh = ds[int(i)]
        r2.append(((x - xh) / ds.sigma_d) ** 2)
    rms = np.sqrt(np.mean(r2))
    assert 0.8 < rms < 1.2


def _prior_copy(tmp_path, **override):
    """The real prior with some arrays replaced, saved to tmp_path."""
    from data_loader.linear_estimate import load_prior
    z = load_prior(PRIOR)
    z.update(override)
    p = tmp_path / 'prior.npz'
    np.savez(p, **z)
    return str(p)


@pytest.mark.dataset
@needs_prior
def test_missing_sigma_d_raises(data_root, tmp_path):
    """No silent fallback: a d the prior has no sigma_d for is an error, not sigma_d = 1."""
    from data_loader.hfd_residual import HFDResidualData
    p = _prior_copy(tmp_path, d_values=np.array([4]), sigma_d=np.array([0.05]))
    with pytest.raises(ValueError, match=r'no sigma_d for d=2; run scripts/compute_residual_scale.py --d_values 2'):
        HFDResidualData(data_path=data_root, split='test', sensor_down_sample_rate=2, prior_path=p)
    assert HFDResidualData(data_path=data_root, split='test', sensor_down_sample_rate=4, prior_path=p).sigma_d == 0.05
    # only the scale script, which is measuring sigma_d, may build the dataset without one
    assert HFDResidualData(data_path=data_root, split='test', sensor_down_sample_rate=2, prior_path=p,
                           require_sigma_d=False).sigma_d is None


@pytest.mark.dataset
@needs_prior
def test_prior_with_a_different_sensor_matrix_is_rejected(data_root, tmp_path):
    from data_loader.hfd_residual import HFDResidualData
    from data_loader.linear_estimate import load_prior
    p = _prior_copy(tmp_path, R=load_prior(PRIOR)['R'] * 1.01)
    with pytest.raises(ValueError, match='sensor matrix'):
        HFDResidualData(data_path=data_root, split='test', sensor_down_sample_rate=4, prior_path=p)


def test_merge_sigma_replaces_one_d_and_keeps_the_others():
    from scripts.compute_residual_scale import merge_sigma
    dv, sg = merge_sigma(np.array([1, 2, 4, 8]), np.array([0.1, 0.2, 0.4, 0.8]), [(4, 0.45), (16, 1.6)])
    assert dv.tolist() == [1, 2, 4, 8, 16] and sg.tolist() == [0.1, 0.2, 0.45, 0.8, 1.6]
    dv, sg = merge_sigma(np.zeros(0, dtype=int), np.zeros(0), [(8, 0.8), (2, 0.2)])   # a freshly fitted prior
    assert dv.tolist() == [2, 8] and sg.tolist() == [0.2, 0.8]
    assert dv.dtype.kind == 'i' and sg.dtype == np.float64
    with pytest.raises(ValueError):
        merge_sigma(np.array([1, 2]), np.array([0.1]), [(4, 0.4)])                      # corrupt input arrays


def test_scale_files_depend_only_on_d_and_seed():
    """sigma_d must not depend on the order of --d_values: each d draws its own files from (seed, d)."""
    from scripts.compute_residual_scale import sample_indices
    a = sample_indices(9000, 500, 4, seed=0)
    assert np.array_equal(a, sample_indices(9000, 500, 4, seed=0))
    assert not np.array_equal(a, sample_indices(9000, 500, 2, seed=0))
    assert len(set(a.tolist())) == 500 and a.max() < 9000
    assert len(sample_indices(30, 500, 4, seed=0)) == 30
