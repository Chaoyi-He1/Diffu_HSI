"""Phase 2a Task 7: every metric agrees with a closed form or a known ordering on synthetic data."""
import numpy as np
import pytest
import torch

from scripts.uq import metrics as M


def test_crps_samples_matches_the_gaussian_closed_form_for_many_samples():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    s = rng.standard_normal((2000, 5000))
    est, exact = M.crps_samples(s, x), M.crps_gaussian(np.zeros(5000), np.ones(5000), x)
    assert est.shape == (5000,) and abs(est.mean() - exact.mean()) / exact.mean() < 0.01
    assert abs(exact.mean() - 1 / np.sqrt(np.pi)) < 0.01                      # CRPS(N(0,1); x ~ N(0,1)) averages to sigma / sqrt(pi) = 0.5642


def test_crps_gaussian_is_the_textbook_formula_and_refuses_nothing():
    v = M.crps_gaussian(np.array([0.0]), np.array([1.0]), np.array([0.0]))
    assert abs(v[0] - (2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))) < 1e-12        # z = 0: 2 phi(0) - 1/sqrt(pi)
    assert M.crps_gaussian(np.array([1.0]), np.array([0.0]), np.array([3.0]))[0] == pytest.approx(2.0, abs=1e-6)   # sigma -> 0: |x - mu|


def test_crps_samples_refuses_fewer_than_two_samples():
    with pytest.raises(ValueError, match='K'):
        M.crps_samples(np.zeros((1, 4)), np.zeros(4))


def test_crps_samples_of_identical_samples_is_the_absolute_error():
    s = np.full((8, 5), 0.25)
    assert np.allclose(M.crps_samples(s, np.full(5, 0.75)), 0.5)


def test_coverage_of_gaussian_samples_is_near_nominal_and_boundaries_count():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(100_000)
    s = rng.standard_normal((32, 100_000))
    cov = M.coverage_samples(s, x)                                                   # K = 32: a fresh draw falls inside the interval
    assert abs(cov[0.5] - 0.5 * 31 / 33) < 0.01 and abs(cov[0.9] - 0.9 * 31 / 33) < 0.01   # spanned by the order statistics w.p. (K-1)/(K+1) x level
    big = M.coverage_samples(rng.standard_normal((512, 20_000)), x[:20_000])        # K = 512: near nominal
    assert abs(big[0.5] - 0.5) < 0.02 and abs(big[0.9] - 0.9) < 0.02
    const = np.zeros((4, 3))
    assert M.coverage_samples(const, np.zeros(3)) == {0.5: 1.0, 0.9: 1.0}            # x == lo == hi is covered
    assert M.coverage_samples(const, np.ones(3)) == {0.5: 0.0, 0.9: 0.0}
    g = M.coverage_gaussian(np.zeros(100_000), np.ones(100_000), x)
    assert abs(g[0.5] - 0.5) < 0.01 and abs(g[0.9] - 0.9) < 0.01


def test_ause_is_zero_for_the_oracle_and_positive_for_a_random_ranking():
    rng = np.random.default_rng(2)
    err = rng.standard_normal(10_000) * rng.uniform(0.1, 2.0, 10_000)
    assert M.ause(np.abs(err), err) == pytest.approx(0.0, abs=1e-12)
    assert M.ause(rng.uniform(size=10_000), err) > 0.05
    curve = M.sparsification_curve(np.abs(err), err ** 2)
    assert curve.shape == (50,) and curve[0] == pytest.approx(1.0) and np.all(np.diff(curve) <= 1e-12)


def test_spread_ratio():
    assert M.spread_ratio(np.full(10, 4.0), np.full(10, 2.0)) == pytest.approx(1.0)
    assert M.spread_ratio(np.full(10, 1.0), np.full(10, 2.0)) == pytest.approx(0.25)


def blurred(u):
    """3 x 3 box blur with reflect padding along the last two axes."""
    p = np.pad(u, [(0, 0)] * (u.ndim - 2) + [(1, 1), (1, 1)], mode='reflect')
    out = np.zeros_like(u)
    for dy in range(3):
        for dx in range(3):
            out += p[..., dy:dy + u.shape[-2], dx:dx + u.shape[-1]]
    return out / 9


def test_hf_energy_ratio_and_grad_w1_detect_blur():
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(4, 32, 32))
    assert M.hf_energy_ratio(x, x) == 0.0 and M.grad_w1(x, x) == 0.0
    b = blurred(x)
    assert M.hf_energy_ratio(b, x) < -0.5 and M.grad_w1(b, x) > 0.01
    assert M.laplacian(np.ones((2, 5, 5))).max() == pytest.approx(0.0)          # a flat image has no Laplacian
    assert M.sobel_mag(np.ones((5, 5))).max() == pytest.approx(0.0)


def test_laplacian_and_sobel_use_reflect_padding_not_zeros():
    u = np.ones((1, 6, 6)) * 0.7
    assert np.abs(M.laplacian(u)).max() < 1e-12 and np.abs(M.sobel_mag(u)).max() < 1e-12


def test_to01_converts_tensors_and_arrays():
    t = torch.tensor([-1.0, 0.0, 1.0])
    assert np.allclose(M.to01(t), [0.0, 0.5, 1.0]) and M.to01(t).dtype == np.float64
    assert np.allclose(M.to01(np.array([-1.0, 1.0])), [0.0, 1.0])
