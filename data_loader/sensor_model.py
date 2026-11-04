"""Sensor response loader on the legacy wavelength axes (numpy + scipy only).

The axes are the legacy ones on purpose (comparability with the paper); the
wavelength / voltage fields stored inside the .mat files are never used.
"""
import hashlib
import os

import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d

DEVICES = ('device1', 'device2', 'ph5')
FILES = {
    'device1': ('R_Device1.mat', 'R'),
    'device2': ('R_Device2.mat', 'R'),
    'ph5': ('PH5_interp_results.mat', 'PH5_interp'),
}


def legacy_axis(device, n_wavelengths):
    if device in ('device1', 'device2'):
        return np.linspace(400, 1000, n_wavelengths)
    if device == 'ph5':
        axis = np.arange(400, 1560, 10)
        if len(axis) != n_wavelengths:
            raise ValueError(f'ph5 legacy axis has {len(axis)} points, matrix has {n_wavelengths}')
        return axis
    raise ValueError(f'unknown device {device!r}; expected one of {DEVICES}')


def raw_matrix(device, data_root):
    """[n_wavelengths, n_states] float64; the Device 2 file stores it transposed."""
    fname, key = FILES[device]
    R = np.array(sio.loadmat(os.path.join(data_root, fname))[key], dtype=np.float64)
    if device == 'device2':
        R = R.T
    return R


def load_sensor(device, data_root, n_states=30, wavelens=np.linspace(451, 855, 31)):
    R = raw_matrix(device, data_root)
    n_total = R.shape[1]
    selected = np.linspace(0, n_total - 1, n_states, dtype=int).tolist()
    axis = legacy_axis(device, R.shape[0])
    f = interp1d(axis, R[:, selected], axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    R_raw = f(wavelens)
    mn, mx = R_raw.min(axis=0), R_raw.max(axis=0)
    R_norm = (R_raw - mn) / (mx - mn + 1e-20)
    return dict(device=device, R=R_norm, R_raw=R_raw, selected_states=selected, n_states_total=n_total,
                axis_rule='legacy', file=FILES[device][0],
                sha256=hashlib.sha256(np.ascontiguousarray(R_norm, dtype=np.float64).tobytes()).hexdigest())
