import numpy as np, pytest
from data_loader import mmse as MM
from data_loader.linear_estimate import expand_matrix, linear_inverse
from data_loader.sensor_forward import mosaic_mask, sigma_from_snr

def _toy(seed=0, L=5, S=3):
    rng = np.random.default_rng(seed); B = rng.standard_normal((L, L)); C = B @ B.T + 0.1 * np.eye(L)
    return rng.standard_normal(L), C, rng.random((L, S))

def test_sequential_posterior_matches_bruteforce_gaussian_conditioning():
    mu, C, R = _toy(); s2 = 0.05
    Syy = R.T @ C @ R + s2 * np.eye(3); Sxy = C @ R
    y = np.random.default_rng(1).standard_normal((4, 3))
    xhat, Sig = MM.posterior_sequential(y, R, mu, C, s2)
    np.testing.assert_allclose(xhat, mu + (y - mu @ R) @ np.linalg.solve(Syy, Sxy.T), atol=1e-9)
    np.testing.assert_allclose(Sig, C - Sxy @ np.linalg.solve(Syy, Sxy.T), atol=1e-9)

def test_per_channel_noise_vector_is_accepted():
    mu, C, R = _toy(); y = np.zeros((2, 3))
    a, _ = MM.posterior_sequential(y, R, mu, C, np.array([0.1, 0.2, 0.3])); b, _ = MM.posterior_sequential(y, R, mu, C, 0.1)
    assert not np.allclose(a, b)

def test_snapshot_posterior_equals_sequential_with_one_row():
    mu, C, R = _toy(); y = np.random.default_rng(2).standard_normal(6); k = np.array([0, 1, 2, 0, 1, 2])
    xs, Sn = MM.posterior_snapshot(y, k, R, mu, C, 0.05)
    for n in range(6):
        xr, Sr = MM.posterior_sequential(y[n:n + 1, None], R[:, k[n]:k[n] + 1], mu, C, 0.05)
        np.testing.assert_allclose(xs[n], xr[0], atol=1e-9); np.testing.assert_allclose(Sn[n], Sr, atol=1e-9)

def test_sigma_to_prior_regulariser_reproduces_linear_inverse(small_R, synthetic_prior):
    mu, C = synthetic_prior['mu'], synthetic_prior['C']                     # trainer-scale prior, as the npz
    s = 3e-7
    from scripts.fit_linear_prior import make_gain
    K = make_gain(C, small_R, s)
    x_tr = np.clip(synthetic_prior['X'][:50], -0.9, 0.9); y_tr = x_tr @ small_R
    ref = linear_inverse(y_tr, mu, K, small_R)                                 # trainer scale, clipped to [-1, 1]
    mu01, C01 = MM.prior01(dict(mu=mu, C=C)); y01 = (y_tr + small_R.sum(0)) / 2
    xhat01, _ = MM.posterior_sequential(y01, small_R, mu01, C01, (s / 2) ** 2)      # C01 = C / 4, so nu = s^2 / 4 reproduces K(s)
    np.testing.assert_allclose(np.clip(2 * xhat01 - 1, -1, 1), ref, atol=1e-4)   # linear_inverse clips its output; a few synthetic pixels exceed 1

def test_posterior_std_grows_as_snr_falls_and_is_finite_at_the_floor(small_R, synthetic_prior):
    mu01, C01 = MM.prior01(dict(mu=synthetic_prior['mu'], C=synthetic_prior['C'])); P_y = 50.0
    stds = [np.sqrt(np.diag(MM.posterior_sequential(np.zeros((1, 30)), small_R, mu01, C01, sigma_from_snr(snr, P_y) ** 2)[1])).mean() for snr in (60, 40, 20)]
    assert np.all(np.isfinite(stds)) and stds[0] < stds[1] < stds[2]

def test_mmse_maps_shapes_scales_and_mask_dispatch(small_R, synthetic_prior):
    prior = dict(mu=synthetic_prior['mu'], C=synthetic_prior['C']); A = expand_matrix(np.linspace(451, 855, 31), 64)
    y = np.random.default_rng(0).random((10, 12, 30)); m = mosaic_mask(10, 12)
    xs, ss = MM.mmse_maps(y, None, small_R, prior, 0.1, A); xm, sm = MM.mmse_maps(y * m, m, small_R, prior, 0.1, A)
    assert xs.shape == ss.shape == xm.shape == sm.shape == (10, 12, 64) and xs.dtype == np.float32
    assert ss.min() > 0 and sm.mean() > ss.mean()                               # one measurement per pixel is less informative

def test_demosaic_recovers_a_linear_ramp_in_the_interior():
    H, W = 20, 24; m = mosaic_mask(H, W); ramp = np.add.outer(np.arange(H), 2 * np.arange(W)).astype(float)
    y_full = np.repeat(ramp[..., None], 30, -1); y_m = y_full * m
    d = MM.demosaic(y_m, m)
    np.testing.assert_allclose(d[5:15, 6:18], y_full[5:15, 6:18], atol=1e-6)

def test_demosaic_then_mmse_runs(small_R, synthetic_prior):
    prior = dict(mu=synthetic_prior['mu'], C=synthetic_prior['C']); A = expand_matrix(np.linspace(451, 855, 31), 64)
    m = mosaic_mask(10, 12); y = np.random.default_rng(0).random((10, 12, 30)) * m
    xh, s = MM.demosaic_then_mmse(y, m, small_R, prior, 0.1, A, np.full(30, 0.01))
    assert xh.shape == (10, 12, 64) and np.isfinite(xh).all() and s.min() > 0

def test_lowrank_snapshot_lift_matches_dense(small_R, synthetic_prior):
    mu01, C01 = MM.prior01(dict(mu=synthetic_prior['mu'], C=synthetic_prior['C'])); A = expand_matrix(np.linspace(451, 855, 31), 64)
    m = mosaic_mask(10, 12).reshape(-1, 30); k = m.argmax(1); y_k = np.random.default_rng(0).random(len(k))
    xd, Sig = MM.posterior_snapshot(y_k, k, small_R, mu01, C01, 0.1)
    xl, Cr, den = MM.posterior_snapshot_lowrank(y_k, k, small_R, mu01, C01, 0.1)
    np.testing.assert_allclose(xl, xd, atol=1e-9)
    md, sd = MM.lift_mean_std(xd, Sig, A); ml, sl = MM.lift_mean_std_lowrank(xl, C01, Cr, den, A)
    np.testing.assert_allclose(ml, md, atol=1e-9); np.testing.assert_allclose(sl, sd, atol=1e-9)

def test_sequential_posterior_refuses_more_channels_than_bands():
    mu, C, R = _toy(L=3, S=5)                                   # S > L: the thin-SVD gain would be wrong
    with pytest.raises(AssertionError, match=r'S <= bands L.*\(3, 5\)'):
        MM.posterior_sequential(np.zeros((2, 5)), R, mu, C, 0.1)


# ----------------------------------------------------------------------------- mmse_maps_general (Phase 3b)
def _setup(small_R, synthetic_prior):
    return dict(mu=synthetic_prior['mu'], C=synthetic_prior['C']), expand_matrix(np.linspace(451, 855, 31), 64)


def test_general_equals_mmse_maps_sequential_and_snapshot(small_R, synthetic_prior):
    prior, A = _setup(small_R, synthetic_prior)
    y = np.random.default_rng(0).random((10, 12, 30))
    ones, nv = np.ones_like(y), np.full_like(y, 0.1 ** 2)
    xs, ss = MM.mmse_maps(y, None, small_R, prior, 0.1, A)
    xg, sg = MM.mmse_maps_general(y, ones, nv, small_R, prior, A)
    assert xg.dtype == sg.dtype == np.float32 and xg.shape == (10, 12, 64)
    np.testing.assert_allclose(xg, xs, atol=1e-5); np.testing.assert_allclose(sg, ss, atol=1e-5)
    m = mosaic_mask(10, 12)
    xm, sm = MM.mmse_maps(y * m, m, small_R, prior, 0.1, A)
    xg, sg = MM.mmse_maps_general(y * m, m, nv, small_R, prior, A)
    np.testing.assert_allclose(xg, xm, atol=1e-5); np.testing.assert_allclose(sg, sm, atol=1e-5)


def test_general_matches_a_dense_per_pixel_reference(small_R, synthetic_prior):
    prior, A = _setup(small_R, synthetic_prior)
    g = np.random.default_rng(1)
    y = g.random((5, 6, 30)); mask = (g.random((5, 6, 30)) < 0.4).astype(np.float32); nv = 0.001 + 0.05 * g.random((5, 6, 30))
    mask[2, 3] = 0                                                              # a pixel with no measured element
    xg, sg = MM.mmse_maps_general(y * mask, mask, nv, small_R, prior, A, chunk=7)   # chunks that do not divide 30 pixels
    mu01, C01 = MM.prior01(prior)
    for i in range(5):
        for j in range(6):
            rows = np.nonzero(mask[i, j])[0]
            if len(rows):
                Rr = small_R[:, rows]; N = np.diag(nv[i, j, rows])
                G = C01 @ Rr @ np.linalg.inv(Rr.T @ C01 @ Rr + N)
                x = mu01 + G @ (y[i, j, rows] - Rr.T @ mu01); S = C01 - G @ Rr.T @ C01
            else:
                x, S = mu01, C01
            np.testing.assert_allclose(xg[i, j], 2 * (A @ x) - 1, atol=1e-5)
            np.testing.assert_allclose(sg[i, j], 2 * np.sqrt(np.maximum(np.diag(A @ S @ A.T), 0)), atol=1e-5)
    np.testing.assert_allclose(xg[2, 3], 2 * (A @ mu01) - 1, atol=1e-6)        # the unmeasured pixel: prior mean and std
    np.testing.assert_allclose(sg[2, 3], 2 * np.sqrt(np.diag(A @ C01 @ A.T)), atol=1e-6)
