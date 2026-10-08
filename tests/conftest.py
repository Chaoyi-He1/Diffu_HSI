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
