import numpy as np

from scripts.sensor import stage0_baselines as S

SNRS = (60.0, 50.0, 40.0, 30.0, 20.0)


def _curve(vals):
    return dict(zip(SNRS, vals))


def test_shape_check_passes_same_shape_and_fails_on_a_kink():
    norm = _curve([0.5, 0.6, 0.9, 1.6, 3.0])
    ok = S.shape_check(norm, {k: 1.3 * v for k, v in norm.items()})
    assert ok['passed'] and ok['monotone_norm'] and ok['monotone_raw'] and ok['max_dev'] < 1e-9
    kink = {k: 1.3 * v for k, v in norm.items()}
    kink[40.0] *= 1.2
    bad = S.shape_check(norm, kink)
    assert bad['max_dev'] > 0.10 and not bad['passed']
    nonmono = _curve([0.5, 0.6, 0.55, 1.6, 3.0])
    r = S.shape_check(nonmono, {k: 1.3 * v for k, v in nonmono.items()})
    assert not r['monotone_norm'] and not r['passed']


def test_principal_angles_identity_and_orthogonal(small_R):
    assert max(S.principal_angles(small_R, small_R)) < 1e-4
    A = np.zeros((31, 3)); A[:3] = np.eye(3)
    B = np.zeros((31, 3)); B[3:6] = np.eye(3)
    C = np.zeros((31, 3)); C[1:4] = np.eye(3)
    assert min(S.principal_angles(A, B)) > 80
    assert max(S.principal_angles(A, C)) > 0


def test_raw_sigma_matches_nominal_snr(small_R):
    rng = np.random.default_rng(0)
    cubes = [rng.uniform(0, 1, (8, 8, 31)) for _ in range(3)]
    R_raw = 1e-3 * small_R
    for snr in SNRS:
        s = S.raw_sigma(R_raw, None, snr, cubes=cubes)
        p = np.mean([np.mean((c @ R_raw) ** 2) for c in cubes])
        assert abs(10 * np.log10(p / s ** 2) - snr) < 0.2
