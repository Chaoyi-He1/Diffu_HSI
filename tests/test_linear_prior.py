import os

import numpy as np
import pytest

from scripts.fit_linear_prior import make_gain, choose_s, fit_prior


def test_make_gain_shape_and_noiseless_recovery(small_R, synthetic_prior):
    mu, C, X = synthetic_prior['mu'], synthetic_prior['C'], synthetic_prior['X']
    K = make_gain(C, small_R, s=1e-6)
    assert K.shape == (31, 30)
    Y = X @ small_R
    Xh = mu + (Y - mu @ small_R) @ K.T
    rel = np.linalg.norm(Xh - X, axis=1) / np.linalg.norm(X, axis=1)
    assert rel.mean() < 0.05          # 6-dim spectra seen through 30 channels are recoverable


def test_make_gain_is_finite_for_tiny_s(small_R, synthetic_prior):
    K = make_gain(synthetic_prior['C'], small_R, s=0.0)
    assert np.isfinite(K).all()


def test_inverse_constant_pixel(small_R, synthetic_prior):
    mu, C = synthetic_prior['mu'], synthetic_prior['C']
    K = make_gain(C, small_R, s=1e-6)
    y = (-np.ones(31)) @ small_R        # a flat patch: every band is -1 after the loader's normalisation
    xh = mu + (y - mu @ small_R) @ K.T
    assert np.isfinite(xh).all()


def test_choose_s_picks_a_candidate(small_R, synthetic_prior):
    rng = np.random.default_rng(2)
    held = synthetic_prior['X'][:2000]
    s = choose_s(synthetic_prior, small_R, held, candidates=[1e-7, 1e-5, 1e-3, 1e-1])
    assert s in [1e-7, 1e-5, 1e-3, 1e-1]


def test_write_prior_keeps_sigma_d_unless_asked(tmp_path):
    """Re-fitting must not wipe the sigma_d table of an existing prior (it costs a scale run to rebuild)."""
    from scripts.fit_linear_prior import write_prior
    p = str(tmp_path / 'prior.npz')
    empty = dict(sigma_d=np.zeros(0), d_values=np.zeros(0, dtype=int))
    assert write_prior(p, dict(mu=np.zeros(3), **empty)) == []                    # fresh file: nothing to keep
    np.savez(p, mu=np.zeros(3), sigma_d=np.array([0.1, 0.4]), d_values=np.array([1, 4]))
    assert write_prior(p, dict(mu=np.ones(3), **empty)) == [1, 4]
    with np.load(p) as z:
        assert z['mu'].tolist() == [1.0, 1.0, 1.0]                                 # the new fit is written
        assert z['d_values'].tolist() == [1, 4] and z['sigma_d'].tolist() == [0.1, 0.4]
    assert write_prior(p, dict(mu=np.ones(3), **empty), overwrite_sigma=True) == []
    with np.load(p) as z:
        assert z['d_values'].size == 0 and z['sigma_d'].size == 0
    assert sorted(os.listdir(tmp_path)) == ['prior.npz']                           # the temp file was renamed away


@pytest.mark.dataset
def test_fit_prior_on_real_files(data_root):
    from data_loader.HFD_dataset import HFD_data
    ds = HFD_data(data_path=data_root, train_mode='image', eval_ratio=0.1, split='train',
                  data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=1)
    prior = fit_prior(ds.img_list, ds.sensor_R_matrix, n_files=20, px_per_file=64, seed=0)
    assert prior['mu'].shape == (31,) and prior['C'].shape == (31, 31)
    assert np.all(np.linalg.eigvalsh(prior['C']) > -1e-8)
