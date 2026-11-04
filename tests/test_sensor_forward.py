import numpy as np, torch
from data_loader import sensor_forward as F
from data_loader.linear_estimate import expand_matrix

def test_mosaic_mask_one_hot_and_tile_layout():
    m = F.mosaic_mask(16, 16)
    assert m.shape == (16, 16, 30) and m.dtype == np.float32
    np.testing.assert_array_equal(m.sum(-1), 1)
    assert m[0, 0, 0] == 1 and m[0, 1, 1] == 1 and m[1, 0, 6] == 1 and m[4, 5, 29] == 1 and m[5, 6, 0] == 1
    assert set(np.argmax(m[:5, :6], -1).ravel()) == set(range(30))      # one full tile holds every channel once

def test_sigma_from_snr():
    assert np.isclose(F.sigma_from_snr(30.0, 50.0), np.sqrt(50.0 / 1000.0))
    assert F.sigma_from_snr(60.0, 1.0) > 0

def test_measure_sequential_is_RT_x_plus_seeded_noise(small_R):
    x = np.random.default_rng(0).random((8, 8, 31))
    y, m = F.measure(x, small_R, 'sequential', 0.0, np.random.default_rng(1))
    np.testing.assert_allclose(y, x @ small_R, atol=1e-6); assert m.min() == 1 and y.dtype == np.float32
    y1, _ = F.measure(x, small_R, 'sequential', 0.1, np.random.default_rng(7)); y2, _ = F.measure(x, small_R, 'sequential', 0.1, np.random.default_rng(7))
    np.testing.assert_array_equal(y1, y2); assert not np.allclose(y1, y)

def test_measure_snapshot_zeroes_unmeasured_entries(small_R):
    x = np.random.default_rng(0).random((16, 16, 31))
    y, m = F.measure(x, small_R, 'snapshot', 0.05, np.random.default_rng(1))
    assert np.all(y[m == 0] == 0) and np.all(y[m == 1] != 0)
    with __import__('pytest').raises(ValueError):
        F.measure(x, small_R, 'bayer', 0.0, np.random.default_rng(0))

def test_measured_snr_matches_nominal(small_R):
    rng = np.random.default_rng(0); X = rng.random((50, 8, 8, 31))
    Y = np.stack([x @ small_R for x in X]); P_y = float((Y ** 2).mean()); sigma = F.sigma_from_snr(30.0, P_y)
    N = np.stack([F.measure(x, small_R, 'sequential', sigma, np.random.default_rng(i))[0] for i, x in enumerate(X)]) - Y
    assert abs(10 * np.log10((Y ** 2).mean() / (N ** 2).mean()) - 30.0) < 0.2

def test_guidance_operator_reproduces_y_from_the_64_band_cube(small_R):
    wl = np.linspace(451, 855, 31); A = expand_matrix(wl, 64)
    x01 = np.random.default_rng(0).random((5, 31)); x64 = (2 * x01 - 1) @ A.T          # trainer-scale 64-band cube
    H, b = F.guidance_operator(small_R, A)
    assert H.shape == (30, 64) and b.shape == (30,)
    np.testing.assert_allclose(x64 @ H.T + b, x01 @ small_R, atol=1e-9)

def test_apply_H_and_adjoint(small_R):
    H = torch.tensor(F.guidance_operator(small_R, expand_matrix(np.linspace(451, 855, 31), 64))[0])
    x = torch.randn(2, 64, 4, 4, dtype=torch.float64); v = torch.randn(2, 30, 4, 4, dtype=torch.float64)
    assert F.apply_H(x, H).shape == (2, 30, 4, 4)
    assert torch.allclose((F.apply_H(x, H) * v).sum(), (x * F.apply_HT(v, H)).sum())
