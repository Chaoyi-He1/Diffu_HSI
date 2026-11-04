import os
import sys
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')


def pytest_configure(config):
    config.addinivalue_line('markers', 'dataset: needs the HFD100 dataset on disk')


def pytest_collection_modifyitems(config, items):
    if os.path.isdir(DATA_ROOT):
        return
    skip = pytest.mark.skip(reason=f'dataset not found at {DATA_ROOT}')
    for item in items:
        if 'dataset' in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope='session')
def data_root():
    return DATA_ROOT


@pytest.fixture(scope='session')
def small_R():
    """A deterministic 31x30 sensor matrix with the real matrix's shape and [0, 1] column scaling."""
    rng = np.random.default_rng(0)
    wl = np.linspace(0, 1, 31)
    centers = np.linspace(0, 1, 30)
    R = np.exp(-((wl[:, None] - centers[None, :]) ** 2) / (2 * 0.08 ** 2))  # smooth, overlapping filters
    R = (R - R.min(0)) / (R.max(0) - R.min(0))
    return R


@pytest.fixture(scope='session')
def synthetic_prior(small_R):
    """Gaussian spectra living on a 6-dim subspace, so the 30-channel sensor identifies them."""
    rng = np.random.default_rng(1)
    basis = rng.standard_normal((31, 6))
    coef = rng.standard_normal((20000, 6))
    X = 0.2 * coef @ basis.T
    mu = X.mean(0)
    C = np.cov(X.T) + 1e-6 * np.eye(31)
    return dict(mu=mu, C=C, X=X)


@pytest.fixture(scope='session')
def fake_sensor_root(tmp_path_factory):
    """Tiny HFD-like root: 12 cubes in 3 classes plus synthetic Device1 / Device2 / PH5 response files
    with the real files' keys, shapes and orientation (Device 2 is stored transposed, [states, wavelengths])."""
    import scipy.io as sio
    root = tmp_path_factory.mktemp('fake_sensor_root')
    rng = np.random.default_rng(2)
    basis = rng.standard_normal((31, 6))
    layout = {'P022': ['a_1', 'a_2', 'b_1', 'b_2'], 'P023': ['c_1', 'c_2', 'c_3', 'd_1'],
              'P024': ['e_1', 'e_2', 'e_3', 'e_4']}
    for cls, names in layout.items():
        d = root / 'MatFlower60' / 'Train' / cls
        d.mkdir(parents=True)
        for nm in names:
            coef = rng.standard_normal((16, 16, 6))
            x = coef @ basis.T
            x = (x - x.min()) / (x.max() - x.min())
            sio.savemat(str(d / f'{nm}.mat'), {'truth': x.astype(np.float32)})

    def gaussians(n_wl, n_states, lo, hi, wl_lo, wl_hi):
        wl = np.linspace(wl_lo, wl_hi, n_wl)
        centers = np.linspace(lo, hi, n_states)
        return np.exp(-((wl[:, None] - centers[None, :]) ** 2) / (2 * 25.0 ** 2)), wl

    R1, wl1 = gaussians(401, 71, 400, 800, 400, 1000)
    sio.savemat(str(root / 'R_Device1.mat'), {'R': R1, 'wavelength': wl1[:, None], 'voltage': np.linspace(-1, 1, 71)[None, :]})
    R2, wl2 = gaussians(400, 42, 400, 800, 400, 1000)
    sio.savemat(str(root / 'R_Device2.mat'), {'R': R2.T.copy(), 'wavelength': wl2[None, :], 'voltage': np.linspace(-1, 1, 42)[None, :]})
    R3, _ = gaussians(116, 101, 400, 1500, 400, 1550)
    sio.savemat(str(root / 'PH5_interp_results.mat'), {'PH5_interp': 1e-7 * R3})
    return str(root)
