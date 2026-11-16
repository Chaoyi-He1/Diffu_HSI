"""Phase 3b degradation scenarios (spec 2026-11-16 Part B): a known change of the forward model at test time.

A scenario is a plain JSON-able dict

    {"name": str,
     "R":      {"kind": "device"} | {"kind": "blend", "alpha": a} | {"kind": "shift", "bands": d}
               | {"kind": "gain", "delta": d, "seed": 0},
     "states": None | S,                    # evenly spaced subset of the columns, round(linspace(0, n - 1, S))
     "pixels": None | {"kind": "dead", "p": p} | {"kind": "stride", "d": d},
     "noise":  {"kind": "gaussian"} | {"kind": "poisson"} | {"kind": "saturation", "q": q}
               | {"kind": "quantisation", "bits": b}}

measure_scenario returns the AWARE measurement (y, a 0/1 mask of the elements used and their noise variance) and, for
the non-Gaussian noise kinds, the NAIVE one a white-Gaussian pipeline at the nominal sigma would use. Everything is on
the physical x01 scale (y = x01 @ R + noise), as data_loader.sensor_forward.measure. numpy / scipy only.
"""
import hashlib
import json

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator

from data_loader.mmse import _to_trainer, lift_mean_std, posterior_sequential, prior01

R_KINDS = ('device', 'blend', 'shift', 'gain')
PIXEL_KINDS = ('dead', 'stride')
NOISE_KINDS = ('gaussian', 'poisson', 'saturation', 'quantisation')
NAIVE_KINDS = ('poisson', 'saturation', 'quantisation')        # also the kinds that need scenario_consts
QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


# ----------------------------------------------------------------------------- flags and identity
def has_naive(scenario):
    return scenario['noise']['kind'] in NAIVE_KINDS


def needs_consts(scenario):
    return scenario['noise']['kind'] in NAIVE_KINDS


def has_pixels(scenario):
    return scenario.get('pixels') is not None


def scenario_sha(scenario):
    """sha256 of the canonical JSON (sorted keys, no whitespace)."""
    blob = json.dumps(scenario, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def state_indices(n, S):
    return np.round(np.linspace(0, n - 1, int(S))).astype(int)


# ----------------------------------------------------------------------------- the response matrix
def _shift(R, bands):
    """Row i takes the value at row i - bands (linear interpolation; the first / last row's value held outside)."""
    n = R.shape[0]
    src = np.arange(n, dtype=np.float64) - float(bands)
    rows = np.arange(n, dtype=np.float64)
    return np.stack([np.interp(src, rows, R[:, k]) for k in range(R.shape[1])], 1)


def scenario_R(R1, R2, spec):
    """R [31, S] float64 of the scenario: R1 = Device 1's response, R2 = Device 2's (only read by 'blend')."""
    R1 = np.asarray(R1, np.float64)
    r = spec['R']
    kind = r['kind']
    if kind == 'device':
        R = R1.copy()
    elif kind == 'blend':
        R2 = np.asarray(R2, np.float64)
        if R2.shape != R1.shape:
            raise ValueError(f'blend needs R1 and R2 of the same shape, got {R1.shape} and {R2.shape}')
        a = float(r['alpha'])
        R = (1.0 - a) * R1 + a * R2
    elif kind == 'shift':
        R = _shift(R1, r['bands'])
    elif kind == 'gain':
        g = np.random.default_rng(int(r.get('seed', 0))).standard_normal(R1.shape[1])
        R = R1 * np.exp(float(r['delta']) * g)[None, :]
    else:
        raise ValueError(f'unknown R kind {kind!r}; expected one of {R_KINDS}')
    if spec.get('states') is not None:
        R = R[:, state_indices(R.shape[1], spec['states'])]
    return np.ascontiguousarray(R)


# ----------------------------------------------------------------------------- pixel patterns
def pixel_measured(H, W, spec, rng):
    """bool [H, W]: which pixels have a measurement. 'dead' draws rng.random((H, W)) (a pixel is measured where the
    draw is >= p; if none is, pixel (0, 0) is); 'stride' and None use no randomness."""
    if spec is None:
        return np.ones((H, W), bool)
    kind = spec['kind']
    if kind == 'dead':
        m = rng.random((H, W)) >= float(spec['p'])
        if not m.any():
            m[0, 0] = True
        return m
    if kind == 'stride':
        d = int(spec['d'])
        return (np.arange(H)[:, None] % d == 0) & (np.arange(W)[None, :] % d == 0)
    raise ValueError(f'unknown pixel kind {kind!r}; expected one of {PIXEL_KINDS}')


# ----------------------------------------------------------------------------- measurement
def measure_scenario(x01_31, R, sigma, scenario, rng, consts):
    """Measure the cube under the scenario. rng order: the pixel pattern first (pixel_measured), then the noise
    (rng.standard_normal of the readings' shape, or rng.poisson for shot noise).

    Returns float32 [H, W, S] arrays y, mask, noise_var (the aware measurement; y is 0 where mask is 0, noise_var > 0
    everywhere) and, when has_naive(scenario), y_naive, mask_naive, noise_var_naive (white Gaussian at sigma).
    consts (scenario_consts) is needed by poisson (mean_y), saturation (quantiles) and quantisation (y_max)."""
    kind = scenario['noise']['kind']
    if kind not in NOISE_KINDS:
        raise ValueError(f'unknown noise kind {kind!r}; expected one of {NOISE_KINDS}')
    if kind in NAIVE_KINDS and consts is None:
        raise ValueError(f'noise kind {kind!r} needs the scenario constants (scenario_consts.json)')
    H, W = x01_31.shape[:2]
    measured = pixel_measured(H, W, scenario.get('pixels'), rng)
    clean = x01_31 @ R
    s2 = float(sigma) ** 2
    ones = np.ones(clean.shape, np.float64)
    naive = None
    if kind == 'gaussian':
        y = clean + sigma * rng.standard_normal(clean.shape)
        mask, nv = ones, np.full(clean.shape, s2)
    elif kind == 'poisson':
        gam = float(consts['mean_y']) / s2
        y = rng.poisson(gam * np.maximum(clean, 0.0)) / gam
        mask, nv = ones, np.maximum(y, 1.0 / gam) / gam
        naive = (y, ones, np.full(clean.shape, s2))
    elif kind == 'saturation':
        z = clean + sigma * rng.standard_normal(clean.shape)
        cap = float(consts['quantiles'][str(scenario['noise']['q'])])
        sat = z >= cap
        mask = (~sat).astype(np.float64)
        y = np.where(sat, 0.0, z)
        nv = np.full(clean.shape, s2)
        naive = (np.minimum(z, cap), ones, np.full(clean.shape, s2))
    else:                                                            # quantisation
        z = clean + sigma * rng.standard_normal(clean.shape)
        y_max = float(consts['y_max'])
        delta = y_max / (2 ** int(scenario['noise']['bits']) - 1)
        y = np.round(np.clip(z, 0.0, y_max) / delta) * delta
        mask, nv = ones, np.full(clean.shape, s2 + delta ** 2 / 12.0)
        naive = (y, ones, np.full(clean.shape, s2))
    pix = measured[..., None].astype(np.float64)
    mask = mask * pix
    out = dict(y=(y * mask).astype(np.float32), mask=mask.astype(np.float32), noise_var=nv.astype(np.float32))
    if naive is not None:
        yn, mn, nvn = naive
        mn = mn * pix
        out.update(y_naive=(yn * mn).astype(np.float32), mask_naive=mn.astype(np.float32),
                   noise_var_naive=nvn.astype(np.float32))
    return out


# ----------------------------------------------------------------------------- constants
def scenario_consts(clean_readings):
    """clean_readings: an array (any shape) or a list of arrays of clean sensor readings -> dict(mean_y, y_max,
    quantiles={str(q): the q-quantile} for q in QUANTILES), pooled over every reading."""
    if isinstance(clean_readings, (list, tuple)):
        v = np.concatenate([np.asarray(a, np.float64).ravel() for a in clean_readings])
    else:
        v = np.asarray(clean_readings, np.float64).ravel()
    return dict(mean_y=float(v.mean()), y_max=float(v.max()),
                quantiles={str(q): float(np.quantile(v, q)) for q in QUANTILES})


def load_consts(path):
    with open(path) as f:
        c = json.load(f)
    missing = [k for k in ('mean_y', 'y_max', 'quantiles') if k not in c]
    if missing:
        raise ValueError(f'{path}: scenario constants are missing {missing}')
    return c


# ----------------------------------------------------------------------------- interpolation baseline (L_interp)
def _rectilinear(measured):
    """(rows, cols) when the measured pixels are exactly a rectilinear grid with >= 2 rows and columns, else None."""
    rows, cols = np.nonzero(measured.any(1))[0], np.nonzero(measured.any(0))[0]
    if len(rows) < 2 or len(cols) < 2:
        return None
    grid = np.zeros_like(measured)
    grid[np.ix_(rows, cols)] = True
    return (rows, cols) if np.array_equal(grid, measured) else None


def interp_readings(y, measured):
    """Per-channel spatial interpolation of the measured pixels' readings onto the full [H, W] grid -> [H, W, S]
    float64. A rectilinear (stride) grid: RegularGridInterpolator, linear, extrapolating (as mmse.demosaic); otherwise
    griddata-style linear interpolation over the Delaunay triangulation of the measured pixels, with a nearest-neighbour
    fill outside their convex hull (or everywhere when fewer than 3 non-collinear pixels are measured)."""
    y = np.asarray(y, np.float64)
    measured = np.asarray(measured, bool)
    H, W, S = y.shape
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pts = np.stack([ii.ravel(), jj.ravel()], -1)
    if measured.all():
        return y.copy()
    rc = _rectilinear(measured)
    if rc is not None:
        rows, cols = rc
        f = RegularGridInterpolator((rows, cols), y[np.ix_(rows, cols)], method='linear', bounds_error=False,
                                    fill_value=None)
        return f(pts).reshape(H, W, S)
    src = np.stack(np.nonzero(measured), -1).astype(np.float64)
    vals = y[measured]                                                   # [N, S]
    out = np.full((H * W, S), np.nan)
    if len(src) >= 3:
        try:
            out = LinearNDInterpolator(src, vals)(pts.astype(np.float64)).reshape(H * W, S)
        except Exception:                                                # collinear points: Qhull refuses
            pass
    bad = np.isnan(out).any(1)
    if bad.any():
        out[bad] = NearestNDInterpolator(src, vals)(pts[bad].astype(np.float64))
    out = out.reshape(H, W, S)
    out[measured] = y[measured]
    return out


def interp_then_mmse(y, measured, R, prior, sigma, A, e2):
    """Sequential posterior of the interpolated readings with per-channel noise_var = sigma^2 + e2 (as
    mmse.demosaic_then_mmse). -> (x_hat64, std64) [H, W, 64] float32 on the trainer scale."""
    H, W, S = y.shape
    mu01, C01 = prior01(prior)
    d = interp_readings(y, measured)
    xh, Sig = posterior_sequential(d.reshape(-1, S), R, mu01, C01, float(sigma) ** 2 + np.asarray(e2, np.float64))
    return _to_trainer(*lift_mean_std(xh, Sig, A), (H, W, A.shape[0]))


def interp_error_e2(cubes01, R, pixels_spec, seed):
    """Mean squared interpolation error of the CLEAN readings per channel, pooled over the unmeasured pixels of the
    cubes -> [S] float64. Cube c's pixel pattern is drawn from np.random.default_rng([seed, c]). Zeros when no pixel
    is unmeasured (pixels_spec None)."""
    R = np.asarray(R, np.float64)
    tot, cnt = np.zeros(R.shape[1]), 0
    for c, x in enumerate(cubes01):
        clean = np.asarray(x, np.float64) @ R
        meas = pixel_measured(clean.shape[0], clean.shape[1], pixels_spec, np.random.default_rng([int(seed), c]))
        if meas.all():
            continue
        err = (interp_readings(clean * meas[..., None], meas) - clean)[~meas]
        tot += (err ** 2).sum(0)
        cnt += err.shape[0]
    return tot / cnt if cnt else tot
