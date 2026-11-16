# tests/test_posterior_draw.py — the last-step Gaussian posterior draw of line B+ against closed forms
import numpy as np
import pytest
import torch

from model import posterior_draw as P

C, O = 6, 4


def _problem(seed=0):
    g = np.random.default_rng(seed)
    H = g.standard_normal((O, C))
    b = g.standard_normal(O) * 0.1
    L = g.standard_normal((C, C)) * 0.2
    S = L @ L.T + 0.01 * np.eye(C)
    s0 = g.standard_normal(C) * 0.2
    y0 = H @ s0 + b + g.standard_normal(O) * 0.3
    return H, b, S, s0, y0


def _tiles(s0, y0, hw):
    s = torch.tensor(s0, dtype=torch.float32)[None, :, None, None].expand(1, C, hw, hw).clone()
    y = torch.tensor(y0, dtype=torch.float32)[None, :, None, None].expand(1, O, hw, hw).clone()
    return s, y


def _moments(x, sel=None):
    v = x[0].permute(1, 2, 0).reshape(-1, C).double().numpy()
    if sel is not None:
        v = v[sel]
    return v.mean(0), np.cov(v.T)


def _closed(H, S, s0, y0, b, nv, rows=None):
    rows = np.arange(O) if rows is None else np.asarray(rows)
    Hr = H[rows]
    N = np.diag(np.broadcast_to(np.asarray(nv, dtype=np.float64), (O,))[rows])
    G = S @ Hr.T @ np.linalg.inv(Hr @ S @ Hr.T + N)
    return s0 + G @ (y0[rows] - b[rows] - Hr @ s0), S - G @ Hr @ S


def test_every_row_measured_matches_the_gaussian_posterior():
    H, b, S, s0, y0 = _problem()
    s, y = _tiles(s0, y0, 90)                                        # 8100 pixels, each an independent draw
    x = P.posterior_draw(s, H, b, y, 0.09, S, seed=1, clamp=None)
    mu, Sp = _closed(H, S, s0, y0, b, 0.09)
    m, cov = _moments(x)
    assert np.allclose(m, mu, atol=0.02) and np.allclose(cov, Sp, atol=0.012)
    assert np.trace(Sp) < np.trace(S)                                # conditioning removes variance


def test_general_path_equals_the_fast_path():
    H, b, S, s0, y0 = _problem(1)
    s, y = _tiles(s0, y0, 7)
    a = P.posterior_draw(s, H, b, y, 0.04, S, seed=3, clamp=None)
    c = P.posterior_draw(s, H, b, y, torch.full_like(y, 0.04), S, seed=3, mask=torch.ones_like(y), clamp=None)
    assert torch.allclose(a, c, atol=1e-5)


def test_one_row_per_pixel_and_unmeasured_pixels():
    H, b, S, s0, y0 = _problem(2)
    hw = 120
    s, y = _tiles(s0, y0, hw)
    mask = torch.zeros(1, O, hw, hw)
    mask[:, 2, :, : hw // 2] = 1.0                                   # left half measures row 2, right half nothing
    x = P.posterior_draw(s, H, b, y * mask, 0.05, S, seed=5, mask=mask, clamp=None)
    left = (mask[0, 2] == 1).reshape(-1).numpy()
    mu, Sp = _closed(H, S, s0, y0, b, 0.05, rows=[2])
    m, cov = _moments(x, left)
    assert np.allclose(m, mu, atol=0.025) and np.allclose(cov, Sp, atol=0.015)
    m, cov = _moments(x, ~left)                                      # no measurement: the kernel alone
    assert np.allclose(m, s0, atol=0.025) and np.allclose(cov, S, atol=0.015)


def test_per_element_noise_variance():
    H, b, S, s0, y0 = _problem(3)
    hw = 120
    s, y = _tiles(s0, y0, hw)
    nv = torch.full((1, O, hw, hw), 0.02)
    nv[:, :, hw // 2:] = 0.5                                         # the lower half is much noisier
    nv[:, 1] *= 3.0                                                  # and row 1 noisier than the others everywhere
    x = P.posterior_draw(s, H, b, y, nv, S, seed=7, clamp=None)
    top = np.zeros((hw, hw), bool); top[: hw // 2] = True
    for sel, base in ((top.reshape(-1), 0.02), (~top.reshape(-1), 0.5)):
        per_row = np.full(O, base); per_row[1] *= 3.0
        mu, Sp = _closed(H, S, s0, y0, b, per_row)
        m, cov = _moments(x, sel)
        assert np.allclose(m, mu, atol=0.03) and np.allclose(cov, Sp, atol=0.02)


def test_seeded_clamped_and_stacked():
    H, b, S, s0, y0 = _problem(4)
    s, y = _tiles(s0, y0, 5)
    a = P.posterior_draw(s, H, b, y, 0.1, S, seed=11)
    assert torch.equal(a, P.posterior_draw(s, H, b, y, 0.1, S, seed=11))
    assert not torch.equal(a, P.posterior_draw(s, H, b, y, 0.1, S, seed=12))
    assert float(a.abs().max()) <= 1.0 and a.dtype == s.dtype
    big = P.posterior_draw(s, H, b, y, 0.1, 400.0 * S, seed=11)       # a huge kernel is clamped to the cube's range
    assert float(big.max()) == 1.0 and float(big.min()) == -1.0
    draws = torch.stack([s, s, s])
    out = P.posterior_draws(draws, H, b, y, 0.1, S)
    assert out.shape == draws.shape
    assert torch.equal(out[1], P.posterior_draw(s, H, b, y, 0.1, S, seed=P.SEED_OFFSET + 1))
    assert not torch.equal(out[0], out[1])


def test_refuses_bad_inputs(tmp_path):
    H, b, S, s0, y0 = _problem(5)
    s, y = _tiles(s0, y0, 3)
    with pytest.raises(ValueError, match='positive'):
        P.posterior_draw(s, H, b, y, 0.0, S, seed=0)
    with pytest.raises(ValueError, match='0/1'):
        P.posterior_draw(s, H, b, y, 0.1, S, seed=0, mask=0.5 * torch.ones_like(y))
    with pytest.raises(ValueError, match='shapes'):
        P.posterior_draw(s, H[:, :-1], b, y, 0.1, S, seed=0)
    bad = tmp_path / 'k.npz'
    np.savez(bad, Sigma_e=-np.eye(C))
    with pytest.raises(ValueError, match='positive semi-definite'):
        P.load_kernel(str(bad))


def test_pooled_error_covariance_and_kernel_roundtrip(tmp_path):
    g = np.random.default_rng(0)
    L = g.standard_normal((C, C)) * 0.3
    r = (g.standard_normal((4, 50, 50, C)) @ L.T + 0.7).transpose(0, 3, 1, 2)     # a constant offset is removed
    S = P.pooled_error_covariance(r)
    assert np.allclose(S, L @ L.T, atol=0.03)
    p = tmp_path / 'kernel.npz'
    np.savez(p, Sigma_e=S, meta=np.array({'n_cubes': 4}, dtype=object))
    S2, meta = P.load_kernel(str(p))
    assert np.array_equal(S, S2) and meta['n_cubes'] == 4 and len(P.kernel_sha256(S2)) == 64
