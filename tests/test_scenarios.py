# tests/test_scenarios.py — the Phase 3b degradation scenarios (data_loader/scenarios.py)
import json

import numpy as np
import pytest

from data_loader import scenarios as SC
from data_loader import mmse as MM
from data_loader.linear_estimate import expand_matrix


def _scen(**kw):
    s = dict(name='t', R={'kind': 'device'}, states=None, pixels=None, noise={'kind': 'gaussian'})
    s.update(kw)
    return s


def _consts():
    return dict(mean_y=2.0, y_max=6.0, quantiles={'0.9': 4.0, '0.7': 3.0})


# ----------------------------------------------------------------------------- the response matrix
def test_scenario_R_device_blend_and_shape_check(small_R):
    R2 = np.flipud(small_R).copy()
    assert np.array_equal(SC.scenario_R(small_R, R2, _scen()), small_R)
    np.testing.assert_allclose(SC.scenario_R(small_R, R2, _scen(R={'kind': 'blend', 'alpha': 0.25})),
                               0.75 * small_R + 0.25 * R2)
    with pytest.raises(ValueError, match='shape'):
        SC.scenario_R(small_R, R2[:, :20], _scen(R={'kind': 'blend', 'alpha': 0.5}))
    with pytest.raises(ValueError, match='kind'):
        SC.scenario_R(small_R, R2, _scen(R={'kind': 'warp'}))


def test_scenario_R_shift_moves_toward_longer_wavelengths(small_R):
    R = SC.scenario_R(small_R, None, _scen(R={'kind': 'shift', 'bands': 2}))
    np.testing.assert_array_equal(R[2:], small_R[:-2])                            # row i takes row i - 2
    np.testing.assert_array_equal(R[:2], np.repeat(small_R[:1], 2, 0))            # the first row's value held
    Rh = SC.scenario_R(small_R, None, _scen(R={'kind': 'shift', 'bands': 0.5}))
    np.testing.assert_allclose(Rh[1:], 0.5 * (small_R[1:] + small_R[:-1]))       # linear interpolation
    np.testing.assert_array_equal(Rh[0], small_R[0])
    # the peak of every column moves by the shift (the fixture's filters peak inside the band range)
    assert np.all(np.argmax(R[:, 5:25], 0) == np.argmax(small_R[:, 5:25], 0) + 2)


def test_scenario_R_gain_is_seeded_per_state(small_R):
    R = SC.scenario_R(small_R, None, _scen(R={'kind': 'gain', 'delta': 0.2, 'seed': 0}))
    g = np.random.default_rng(0).standard_normal(30)
    np.testing.assert_allclose(R, small_R * np.exp(0.2 * g)[None, :])
    assert not np.allclose(R, SC.scenario_R(small_R, None, _scen(R={'kind': 'gain', 'delta': 0.2, 'seed': 1})))


def test_scenario_R_state_subset(small_R):
    R = SC.scenario_R(small_R, None, _scen(states=5))
    idx = np.round(np.linspace(0, 29, 5)).astype(int)
    assert R.shape == (31, 5) and np.array_equal(R, small_R[:, idx]) and list(idx) == [0, 7, 14, 22, 29]
    Rg = SC.scenario_R(small_R, None, _scen(R={'kind': 'gain', 'delta': 0.5, 'seed': 0}, states=10))
    np.testing.assert_allclose(Rg, SC.scenario_R(small_R, None, _scen(R={'kind': 'gain', 'delta': 0.5, 'seed': 0}))[
        :, np.round(np.linspace(0, 29, 10)).astype(int)])                         # gain over the 30 states, then the subset


# ----------------------------------------------------------------------------- pixel patterns
def test_pixel_measured_patterns():
    rng = np.random.default_rng(0)
    assert SC.pixel_measured(4, 5, None, rng).all()
    st = SC.pixel_measured(9, 10, {'kind': 'stride', 'd': 4}, rng)
    ii, jj = np.nonzero(st)
    assert set(ii) == {0, 4, 8} and set(jj) == {0, 4, 8} and st.sum() == 9
    d = SC.pixel_measured(200, 200, {'kind': 'dead', 'p': 0.9}, np.random.default_rng(1))
    assert abs(d.mean() - 0.1) < 0.01
    np.testing.assert_array_equal(d, np.random.default_rng(1).random((200, 200)) >= 0.9)
    none = SC.pixel_measured(3, 3, {'kind': 'dead', 'p': 1.0}, np.random.default_rng(0))
    assert none.sum() == 1 and none[0, 0]                                        # at least one pixel is measured


# ----------------------------------------------------------------------------- measurements
def test_measure_gaussian_matches_sensor_forward(small_R):
    from data_loader.sensor_forward import measure
    x = np.random.default_rng(0).random((6, 7, 31))
    m = SC.measure_scenario(x, small_R, 0.1, _scen(), np.random.default_rng(5), None)
    y_ref, _ = measure(x, small_R, 'sequential', 0.1, np.random.default_rng(5))
    np.testing.assert_array_equal(m['y'], y_ref)                                 # same rng stream as the Phase 3 forward model
    assert set(m) == {'y', 'mask', 'noise_var'} and m['mask'].min() == 1
    assert all(v.dtype == np.float32 and v.shape == (6, 7, 30) for v in m.values())
    np.testing.assert_allclose(m['noise_var'], 0.01, rtol=1e-6)


def test_measure_with_pixels_draws_the_pattern_first(small_R):
    x = np.random.default_rng(0).random((10, 10, 31))
    sc = _scen(pixels={'kind': 'dead', 'p': 0.5})
    m = SC.measure_scenario(x, small_R, 0.1, sc, np.random.default_rng(3), None)
    rng = np.random.default_rng(3)
    meas = SC.pixel_measured(10, 10, sc['pixels'], rng)
    y = x @ small_R + 0.1 * rng.standard_normal((10, 10, 30))
    np.testing.assert_array_equal(m['mask'], np.repeat(meas[..., None], 30, -1).astype(np.float32))
    np.testing.assert_allclose(m['y'], (y * meas[..., None]).astype(np.float32))
    assert np.all(m['y'][~meas] == 0) and np.array_equal(m['measured'], meas) and m['measured'].dtype == bool


def test_measure_poisson(small_R):
    x = np.random.default_rng(0).random((40, 40, 31))
    c = _consts(); sigma = 0.3
    m = SC.measure_scenario(x, small_R, sigma, _scen(noise={'kind': 'poisson'}), np.random.default_rng(1), c)
    gam = c['mean_y'] / sigma ** 2
    clean = x @ small_R
    k = m['y'] * gam
    np.testing.assert_allclose(k, np.round(k), atol=1e-3)                           # counts / gamma
    assert abs((m['y'] - clean).mean()) < 0.01 and abs(((m['y'] - clean) ** 2).mean() / (clean / gam).mean() - 1) < 0.05
    np.testing.assert_allclose(m['noise_var'], np.maximum(m['y'], 1 / gam) / gam, rtol=1e-5)
    np.testing.assert_array_equal(m['y_naive'], m['y']) and np.testing.assert_array_equal(m['mask_naive'], m['mask'])
    np.testing.assert_allclose(m['noise_var_naive'], sigma ** 2, rtol=1e-6)
    assert m['noise_var'].min() > 0 and SC.has_naive(_scen(noise={'kind': 'poisson'}))


def test_measure_saturation_drops_the_clipped_readings_when_aware(small_R):
    x = np.random.default_rng(0).random((12, 12, 31))
    c = _consts(); sigma = 0.1
    sc = _scen(noise={'kind': 'saturation', 'q': 0.7})
    m = SC.measure_scenario(x, small_R, sigma, sc, np.random.default_rng(2), c)
    z = x @ small_R + sigma * np.random.default_rng(2).standard_normal((12, 12, 30))
    sat = z >= 3.0
    assert sat.any() and (~sat).any()
    np.testing.assert_array_equal(m['mask'], (~sat).astype(np.float32))
    np.testing.assert_allclose(m['y'], np.where(sat, 0.0, z).astype(np.float32))
    np.testing.assert_allclose(m['y_naive'], np.minimum(z, 3.0).astype(np.float32))
    assert m['mask_naive'].min() == 1
    np.testing.assert_allclose(m['noise_var'], sigma ** 2, rtol=1e-6)


def test_measure_quantisation(small_R):
    x = np.random.default_rng(0).random((12, 12, 31))
    c = _consts(); sigma = 0.05
    m = SC.measure_scenario(x, small_R, sigma, _scen(noise={'kind': 'quantisation', 'bits': 4}), np.random.default_rng(4), c)
    delta = 6.0 / 15
    np.testing.assert_allclose(m['y'] / delta, np.round(m['y'] / delta), atol=1e-4)
    assert m['y'].min() >= 0 and m['y'].max() <= 6.0 + 1e-6
    np.testing.assert_allclose(m['noise_var'], sigma ** 2 + delta ** 2 / 12, rtol=1e-6)
    np.testing.assert_allclose(m['noise_var_naive'], sigma ** 2, rtol=1e-6)
    np.testing.assert_array_equal(m['y_naive'], m['y'])


def test_naive_masks_carry_the_pixel_pattern(small_R):
    x = np.random.default_rng(0).random((10, 10, 31))
    sc = _scen(pixels={'kind': 'stride', 'd': 4}, noise={'kind': 'saturation', 'q': 0.9})
    m = SC.measure_scenario(x, small_R, 0.1, sc, np.random.default_rng(0), _consts())
    st = SC.pixel_measured(10, 10, sc['pixels'], None)
    assert np.all(m['mask_naive'][~st] == 0) and np.all(m['mask_naive'][st] == 1) and np.all(m['y_naive'][~st] == 0)
    assert np.all(m['mask'][~st] == 0)


def test_flags_sha_and_unknown_noise(small_R):
    assert not SC.has_naive(_scen()) and not SC.has_pixels(_scen()) and SC.has_pixels(_scen(pixels={'kind': 'stride', 'd': 4}))
    a, b = _scen(), dict(reversed(list(_scen().items())))
    assert SC.scenario_sha(a) == SC.scenario_sha(b) and len(SC.scenario_sha(a)) == 64
    assert SC.scenario_sha(a) != SC.scenario_sha(_scen(states=5))
    with pytest.raises(ValueError, match='noise'):
        SC.measure_scenario(np.zeros((2, 2, 31)), small_R, 0.1, _scen(noise={'kind': 'laplace'}), np.random.default_rng(0), None)
    with pytest.raises(ValueError, match='constants'):
        SC.measure_scenario(np.zeros((2, 2, 31)), small_R, 0.1, _scen(noise={'kind': 'poisson'}), np.random.default_rng(0), None)


def test_consts_roundtrip(tmp_path):
    y = np.random.default_rng(0).random((5, 6, 30)) * 3
    c = SC.scenario_consts([y[:2], y[2:]])
    assert np.isclose(c['mean_y'], y.mean()) and np.isclose(c['y_max'], y.max())
    assert np.isclose(c['quantiles']['0.9'], np.quantile(y, 0.9)) and np.isclose(c['quantiles']['0.7'], np.quantile(y, 0.7))
    p = tmp_path / 'c.json'
    p.write_text(json.dumps(dict(c, files=['a'])))
    assert SC.load_consts(str(p))['quantiles'] == c['quantiles']


# ----------------------------------------------------------------------------- interpolation baseline
def test_interp_readings_stride_matches_demosaic_rule_and_recovers_a_ramp():
    Hh, W = 17, 21
    ramp = np.add.outer(np.arange(Hh), 2 * np.arange(W)).astype(float)
    y = np.repeat(ramp[..., None], 3, -1)
    st = SC.pixel_measured(Hh, W, {'kind': 'stride', 'd': 4}, None)
    d = SC.interp_readings(y * st[..., None], st)
    np.testing.assert_allclose(d, y, atol=1e-9)                                     # linear, extrapolating
    dead = SC.pixel_measured(Hh, W, {'kind': 'dead', 'p': 0.6}, np.random.default_rng(0))
    dd = SC.interp_readings(y * dead[..., None], dead)
    assert np.isfinite(dd).all()
    np.testing.assert_array_equal(dd[dead], y[dead])                                # data points kept
    inside = np.zeros_like(dead)
    inside[4:-4, 4:-4] = True
    np.testing.assert_allclose(dd[inside], y[inside], atol=1e-6)                    # linear inside the hull


def test_interp_then_mmse_and_e2(small_R, synthetic_prior):
    prior = dict(mu=synthetic_prior['mu'], C=synthetic_prior['C']); A = expand_matrix(np.linspace(451, 855, 31), 64)
    rng = np.random.default_rng(0)
    cubes = [np.clip(0.5 + synthetic_prior['X'][rng.integers(0, 20000, 144)].reshape(12, 12, 31), 0, 1) for _ in range(3)]
    e2 = SC.interp_error_e2(cubes, small_R, {'kind': 'stride', 'd': 4}, 0)
    assert e2.shape == (30,) and (e2 > 0).all()
    assert np.array_equal(SC.interp_error_e2(cubes, small_R, None, 0), np.zeros(30))
    st = SC.pixel_measured(12, 12, {'kind': 'stride', 'd': 4}, None)
    y = (cubes[0] @ small_R) * st[..., None]
    xh, s = SC.interp_then_mmse(y, st, small_R, prior, 0.05, A, e2)
    mu01, C01 = MM.prior01(prior)
    xr, Sr = MM.posterior_sequential(SC.interp_readings(y, st).reshape(-1, 30), small_R, mu01, C01, 0.05 ** 2 + e2)
    x64, s64 = MM.lift_mean_std(xr, Sr, A)
    np.testing.assert_allclose(xh, (2 * x64 - 1).reshape(12, 12, 64), atol=1e-5)
    np.testing.assert_allclose(s, (2 * s64).reshape(12, 12, 64), atol=1e-5)
    assert xh.dtype == np.float32 and s.dtype == np.float32
