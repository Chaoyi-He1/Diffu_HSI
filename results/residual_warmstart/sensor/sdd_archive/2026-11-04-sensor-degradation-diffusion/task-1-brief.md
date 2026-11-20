### Task 1: Sensor response loader on the legacy axes

**Files:**
- Create: `data_loader/sensor_model.py`
- Modify: `tests/conftest.py` (add the `fake_sensor_root` fixture)
- Test: `tests/test_sensor_model.py`

**Interfaces:**
- Produces: `DEVICES = ('device1', 'device2', 'ph5')`; `FILES = {'device1': ('R_Device1.mat', 'R'), 'device2': ('R_Device2.mat', 'R'), 'ph5': ('PH5_interp_results.mat', 'PH5_interp')}`; `def legacy_axis(device: str, n_wavelengths: int) -> np.ndarray` (device1 / device2: `np.linspace(400, 1000, n)`; ph5: `np.arange(400, 1560, 10)`, which must have length n or raise ValueError); `def raw_matrix(device: str, data_root: str) -> np.ndarray` ([n_wavelengths, n_states] float64; device2 transposed); `def load_sensor(device: str, data_root: str, n_states: int = 30, wavelens: np.ndarray = np.linspace(451, 855, 31)) -> dict` with keys `device, R (31 × 30 normalised), R_raw (31 × 30, same axis and states, un-normalised), selected_states (list[int]), n_states_total, axis_rule ('legacy'), file, sha256` where `sha256 = hashlib.sha256(np.ascontiguousarray(R, dtype=np.float64).tobytes()).hexdigest()`.
- Fixture `fake_sensor_root(tmp_path_factory)`: a directory with `MatFlower60/Train/P022/{a_1,a_2,b_1,b_2}.mat`, `P023/{c_1,c_2,c_3,d_1}.mat`, `P024/{e_1,e_2,e_3,e_4}.mat` (12 cubes, `truth` [16, 16, 31] float32 in [0, 1], smooth spectra on a 6-dim subspace like `synthetic_prior`), `R_Device1.mat` (`R` [401 × 71] Gaussian filters over 400–800 nm, `wavelength`, `voltage`), `R_Device2.mat` (`R` stored [42 × 400], i.e. transposed), `PH5_interp_results.mat` (`PH5_interp` [116 × 101], values of order 1e-7).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sensor_model.py
import numpy as np, pytest, scipy.io as sio
from data_loader import sensor_model as SM

def test_device2_file_is_transposed_before_use(fake_sensor_root):
    assert SM.raw_matrix('device2', fake_sensor_root).shape == (400, 42)
    assert SM.raw_matrix('device1', fake_sensor_root).shape == (401, 71)

def test_legacy_axes():
    np.testing.assert_allclose(SM.legacy_axis('device1', 401)[[0, -1]], [400, 1000])
    assert len(SM.legacy_axis('ph5', 116)) == 116 and SM.legacy_axis('ph5', 116)[0] == 400
    with pytest.raises(ValueError):
        SM.legacy_axis('ph5', 100)

def test_load_sensor_shapes_selection_and_normalisation(fake_sensor_root):
    s = SM.load_sensor('device1', fake_sensor_root)
    assert s['R'].shape == (31, 30) and s['R_raw'].shape == (31, 30)
    assert s['selected_states'] == np.linspace(0, 70, 30, dtype=int).tolist() and s['n_states_total'] == 71
    np.testing.assert_allclose(s['R'].min(0), 0, atol=1e-12); np.testing.assert_allclose(s['R'].max(0), 1, atol=1e-12)
    assert s['axis_rule'] == 'legacy' and len(s['sha256']) == 64
    assert SM.load_sensor('ph5', fake_sensor_root)['R'].shape == (31, 30)

def test_sha_changes_with_device(fake_sensor_root):
    assert SM.load_sensor('device1', fake_sensor_root)['sha256'] != SM.load_sensor('device2', fake_sensor_root)['sha256']

@pytest.mark.dataset
def test_device1_equals_legacy_loader(data_root):
    from data_loader.HFD_dataset import HFD_data
    ds = HFD_data(data_path=data_root, train_mode='image', eval_ratio=0.1, split='test', data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=1)
    np.testing.assert_allclose(SM.load_sensor('device1', data_root)['R'], ds.sensor_R_matrix, atol=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_sensor_model.py -q`
Expected: FAIL with `ModuleNotFoundError: data_loader.sensor_model` (and the fixture missing).

- [ ] **Step 3: Add the `fake_sensor_root` fixture to `tests/conftest.py`** (session scope; writes the 12 cubes and the three .mat files described in Interfaces; Device 1 filters: 71 Gaussians with centres spread over 400–800 nm and width 25 nm; Device 2: 42 filters, stored transposed; PH5: 101 filters × 1e-7).

- [ ] **Step 4: Implement `data_loader/sensor_model.py`** with the functions of Interfaces. `load_sensor`: `raw_matrix` → select states → `scipy.interpolate.interp1d(legacy_axis(...), R_sel, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')(wavelens)` → `R_raw`; `R` = per-column min–max of `R_raw` (`+ 1e-20` in the denominator as the legacy loader).

- [ ] **Step 5: Run the tests to verify they pass** (same command; the dataset test runs on this machine). Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add data_loader/sensor_model.py tests/conftest.py tests/test_sensor_model.py
git commit -m "feat(sensor): response loader on the legacy wavelength axes for device1 / device2 (transposed file) / ph5, normalised and raw R with sha (Phase 3 Task 1)"
```

