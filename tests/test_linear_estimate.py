import numpy as np
import pytest

from data_loader.linear_estimate import expand_matrix, linear_inverse, lift_to_grid, estimate_from_sensor


def test_expand_matrix_matches_loader_interpolation():
    from scipy.interpolate import interp1d
    wl = np.linspace(451, 855, 31)
    A = expand_matrix(wl, 64)
    assert A.shape == (64, 31)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(31)
    ref = interp1d(wl, x, kind='linear', bounds_error=False, fill_value='extrapolate')(np.linspace(451, 855, 64))
    assert np.allclose(A @ x, ref, atol=1e-12)


def test_linear_inverse_is_clipped(small_R, synthetic_prior):
    from scripts.fit_linear_prior import make_gain
    mu, C = synthetic_prior['mu'], synthetic_prior['C']
    K = make_gain(C, small_R, 1e-6)
    y = (3.0 * np.ones(31)) @ small_R           # far outside [-1, 1]
    xh = linear_inverse(y[None, :], mu, K, small_R)
    assert xh.shape == (1, 31) and xh.max() <= 1.0 and xh.min() >= -1.0


def test_lift_identity_d1():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 64, 3))
    assert np.array_equal(lift_to_grid(x, 1, 64, 64), x)


def test_lift_reproduces_ramp_exactly():
    H = W = 64
    ramp = np.arange(H, dtype=np.float64)[:, None].repeat(W, 1)[..., None]   # linear in rows
    for d in (2, 4, 8):
        lo = ramp[::d, ::d]
        up = lift_to_grid(lo, d, H, W)
        last = (lo.shape[0] - 1) * d
        assert np.allclose(up[:last + 1], ramp[:last + 1])         # exact between samples
        assert np.allclose(up[last:], ramp[last])                   # edge-extended after the last sample


def test_lift_non_divisible():
    H = W = 64
    rng = np.random.default_rng(1)
    x = rng.standard_normal((H, W, 2))
    for d in (3, 6):
        lo = x[::d, ::d]                                            # ceil(64/d) samples
        up = lift_to_grid(lo, d, H, W)
        assert up.shape == (H, W, 2)
        assert np.allclose(up[::d, ::d], lo)                        # samples land back on the grid


def test_estimate_from_sensor_shape(small_R, synthetic_prior):
    from scripts.fit_linear_prior import make_gain
    prior = dict(mu=synthetic_prior['mu'], K=make_gain(synthetic_prior['C'], small_R, 1e-6), R=small_R)
    A = expand_matrix()
    y_d = np.zeros((16, 16, 30))
    out = estimate_from_sensor(y_d, prior, 4, 64, 64, A)
    assert out.shape == (64, 64, 64) and np.isfinite(out).all()
