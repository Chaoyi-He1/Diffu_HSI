### Task 4: The forward-model config — P_y, demosaic error table, training sha

**Files:**
- Create: `data_loader/sensor_config.py`, `scripts/sensor/__init__.py`, `scripts/sensor/make_config.py`
- Test: `tests/test_sensor_config.py`

**Interfaces:**
- Consumes: `sensor_model.load_sensor`, `sensor_forward.measure` / `mosaic_mask`, `mmse.demosaic`, `warmstart_paths.{DATA_ROOT, RESULTS_ROOT, PRIOR, file_sha256}`, `HFD_data` (file list of the training split), `fit_linear_prior.load_cube`'s normalisation (re-implemented in Task 5's `load_cube01`; here `make_config` uses `HFDSensorSim.load_cube01` — so **Task 4's script imports from Task 5's module lazily** (`from data_loader.hfd_sensor_sim import load_cube01` inside the function that loops over cubes, never at module level, so Task 4's tests pass before Task 5 exists); finish Task 5 before running `make_config.py` for real).
- Produces: `SENSOR_DIR = os.path.join(RESULTS_ROOT, 'sensor')`; `CONFIG_PATH = os.path.join(SENSOR_DIR, 'config.json')`; `TRAIN_KEYS = ('prior_sha256', 'axis_rule', 'normalisation', 'state_selection', 'P_y', 's_floor01', 'snr_floor_db', 'family', 'device_train', 'tile', 'seed')`; `ALL_KEYS = TRAIN_KEYS + ('devices', 'e2_demosaic', 'n_files_P_y', 'created', 'prior_path')`; `def load_config(path: str = CONFIG_PATH) -> dict` (raises FileNotFoundError / ValueError on missing keys via `check_config`); `def check_config(cfg: dict) -> None`; `def training_sha(cfg: dict) -> str` (sha256 of `json.dumps({k: cfg[k] for k in TRAIN_KEYS}, sort_keys=True, separators=(',', ':'))`); `def default_training_fields(prior_sha: str) -> dict` = `dict(prior_sha256=prior_sha, axis_rule='legacy', normalisation='column-minmax', state_selection='linspace30', s_floor01=0.01, snr_floor_db=60.0, family=dict(snr_low_db=20.0, snr_high_db=50.0, p_floor=0.1, p_snapshot=0.5), device_train='device1', tile=[5, 6], seed=0)` (P_y added by the script).
- `scripts/sensor/make_config.py` CLI: `--out` (default CONFIG_PATH), `--data_root`, `--prior`, `--n_e2_files 200`, `--limit_files 0` (debug: P_y over the first N training cubes), `--seed 0`; computes `P_y` = mean of y² (physical, sequential, device1) over the training split's cubes in float64, `e2_demosaic` [30] = per-channel mean squared error of `demosaic(measure_snapshot(x))` against the clean sequential y over the first `n_e2_files` training cubes (noise-free), `devices` = {name: {sha256, file, n_states_total, selected_states}} for the three devices; refuses to overwrite `--out`; prints P_y and training_sha.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sensor_config.py
import json, pytest
from data_loader import sensor_config as SC

def _cfg(**kw):
    c = SC.default_training_fields('p' * 64); c.update(P_y=50.0, devices={}, e2_demosaic=[0.0] * 30, n_files_P_y=10, created='t', prior_path='x'); c.update(kw); return c

def test_training_sha_ignores_non_training_keys_and_key_order():
    a, b = _cfg(created='t1', n_files_P_y=10), _cfg(created='t2', n_files_P_y=9000)
    assert SC.training_sha(a) == SC.training_sha(b)
    assert SC.training_sha(dict(reversed(list(a.items())))) == SC.training_sha(a)

def test_training_sha_changes_with_P_y_and_family():
    assert SC.training_sha(_cfg(P_y=50.0)) != SC.training_sha(_cfg(P_y=51.0))
    f = dict(_cfg()['family'], p_snapshot=0.4); assert SC.training_sha(_cfg(family=f)) != SC.training_sha(_cfg())

def test_check_config_refuses_missing_keys(tmp_path):
    c = _cfg(); del c['P_y']
    with pytest.raises(ValueError, match='P_y'):
        SC.check_config(c)
    p = tmp_path / 'c.json'; p.write_text(json.dumps(_cfg())); assert SC.load_config(str(p))['s_floor01'] == 0.01
    with pytest.raises(FileNotFoundError):
        SC.load_config(str(tmp_path / 'none.json'))

def test_make_config_refuses_overwrite(tmp_path, monkeypatch):
    from scripts.sensor import make_config as MC
    p = tmp_path / 'c.json'; p.write_text('{}')
    with pytest.raises(FileExistsError):
        MC.main(['--out', str(p), '--data_root', str(tmp_path), '--prior', 'none.npz'])
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, module missing.

- [ ] **Step 3: Implement `data_loader/sensor_config.py` and `scripts/sensor/make_config.py`** per Interfaces (`make_config.main(argv=None)` checks `--out` first, before loading anything).

- [ ] **Step 4: Run to verify pass.** Expected: 4 passed.

- [ ] **Step 5: Commit** — `git add data_loader/sensor_config.py scripts/sensor/__init__.py scripts/sensor/make_config.py tests/test_sensor_config.py` ; `git commit -m "feat(sensor): forward-model config with a training sha over the fields a checkpoint must match; make_config computes P_y and the demosaic error table (Phase 3 Task 4)"`.

