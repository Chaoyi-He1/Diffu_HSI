### Task 5: `HFDSensorSim` — the family / fixed-setting dataset and its collate

**Files:**
- Create: `data_loader/hfd_sensor_sim.py`
- Test: `tests/test_hfd_sensor_sim.py`

**Interfaces:**
- Consumes: `HFD_data` (file list and split only), `sensor_model.load_sensor`, `sensor_forward.{measure, sigma_from_snr, mosaic_mask}`, `mmse.mmse_maps`, `linear_estimate.{expand_matrix, load_prior}`, `sensor_config.{training_sha, check_config}`.
- Produces: `def load_cube01(path: str) -> np.ndarray` ([H, W, 31] float64 = (g − g.min()) / (g.max() − g.min() + 1e-20), the loader's per-cube normalisation on [0, 1]); `MODES_FWD = ('sequential', 'snapshot')`; `class HFDSensorSim(torch.utils.data.Dataset)` with `__init__(self, data_root: str, split: str, config: dict, device: str = 'device1', setting: dict | None = None, prior_path: str = PRIOR, eval_ratio: float = 0.1, type: str = 'Flower')` — `setting=None` → training family; `setting=dict(mode=..., snr_db=...)` → fixed; attributes `data_root`, `prior_path`, `split`, `setting`, `config`, `img_list` (from an internal `HFD_data(..., R_n=1, sensor_down_sample_rate=1)`, settable), `sensor` (the `load_sensor` dict), `R`, `prior`, `A`, `wavelens`, `d = 1`, `sigma_d = 2 * config['s_floor01']`, `s_floor_tr = 2 * config['s_floor01']`, `P_y`, `epoch = 0`, `collate_fn = sensor_collate_fn`, `config_sha = training_sha(config)`; methods `set_epoch(self, e: int)`, `draw_setting(self, idx: int) -> tuple[str, float]` (rng = `np.random.default_rng([config['seed'], self.epoch, idx])`: snapshot with p_snapshot, SNR = floor with p_floor else U[low, high]; fixed setting returns it), `__getitem__(self, idx) -> dict` with keys `x` [H, W, 64] float32 trainer scale (= A applied to 2 x01 − 1), `y` [H, W, 30] float32 physical, `x0_hat` [H, W, 64] float32 trainer scale, `s_post` [H, W, 64] float32 = max(s_tr, s_floor_tr), `s_raw` [H, W, 64] float32 (unfloored, trainer scale), `mask` [H, W, 30] float32, `sigma` float32, `snr_db` float32, `snapshot` float32 (0/1); noise rng = `np.random.default_rng([config['seed'] + 1, self.epoch, idx])`; `checkpoint_extra(self) -> dict(sensor_config_sha, s_floor01, P_y, snr_floor_db, family, device_train, sensor_device, R_sha256)`; `def sensor_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]` (HWC → BCHW for the map keys, 1-D tensors for the scalars).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hfd_sensor_sim.py
import numpy as np, pytest, torch
from data_loader import hfd_sensor_sim as DS
from data_loader.sensor_config import default_training_fields
from data_loader.sensor_forward import mosaic_mask

def _cfg(): return dict(default_training_fields('p' * 64), P_y=50.0)

@pytest.fixture
def ds(fake_sensor_root, synthetic_prior, tmp_path):
    np.savez(tmp_path / 'prior.npz', mu=synthetic_prior['mu'], C=synthetic_prior['C'], s=3e-7)
    return DS.HFDSensorSim(fake_sensor_root, 'train', _cfg(), device='device1', prior_path=str(tmp_path / 'prior.npz'))

def test_item_keys_shapes_and_scales(ds):
    it = ds[0]
    assert set(it) == {'x', 'y', 'x0_hat', 's_post', 's_raw', 'mask', 'sigma', 'snr_db', 'snapshot'}
    assert it['x'].shape == it['x0_hat'].shape == it['s_post'].shape == (16, 16, 64) and it['y'].shape == it['mask'].shape == (16, 16, 30)
    assert it['x'].dtype == np.float32 and -1 <= it['x'].min() and it['x'].max() <= 1
    assert it['s_post'].min() >= 2 * 0.01 - 1e-6 and (it['s_raw'] <= it['s_post'] + 1e-6).all()

def test_y_is_R_transpose_x01_plus_seeded_noise(ds):
    ds_fixed = DS.HFDSensorSim(ds.data_root, 'train', _cfg(), setting=dict(mode='sequential', snr_db=60.0), prior_path=ds.prior_path)
    it = ds_fixed[1]; x01 = DS.load_cube01(ds_fixed.img_list[1]); clean = x01 @ ds_fixed.R
    assert np.abs(it['y'] - clean).max() < 6 * it['sigma'] + 1e-6 and not np.allclose(it['y'], clean)
    np.testing.assert_array_equal(ds_fixed[1]['y'], it['y'])                      # same epoch, same idx -> same noise

def test_set_epoch_changes_training_noise_but_not_the_cube(ds):
    a = ds[2]; ds.set_epoch(1); b = ds[2]
    np.testing.assert_array_equal(a['x'], b['x']); assert not np.array_equal(a['y'], b['y'])

def test_family_frequencies(ds):
    draws = [ds.draw_setting(i) for i in range(2000)]
    assert abs(np.mean([m == 'snapshot' for m, _ in draws]) - 0.5) < 0.04
    assert abs(np.mean([s == 60.0 for _, s in draws]) - 0.1) < 0.03
    others = [s for _, s in draws if s != 60.0]; assert min(others) >= 20.0 and max(others) <= 50.0

def test_snapshot_item_mask_and_zero_off_mask(fake_sensor_root, ds):
    d2 = DS.HFDSensorSim(fake_sensor_root, 'train', _cfg(), setting=dict(mode='snapshot', snr_db=30.0), prior_path=ds.prior_path)
    it = d2[0]; np.testing.assert_array_equal(it['mask'], mosaic_mask(16, 16)); assert np.all(it['y'][it['mask'] == 0] == 0) and it['snapshot'] == 1

def test_collate_and_extra(ds):
    b = DS.sensor_collate_fn([ds[0], ds[1]])
    assert b['x'].shape == (2, 64, 16, 16) and b['y'].shape == (2, 30, 16, 16) and b['sigma'].shape == (2,)
    e = ds.checkpoint_extra(); assert e['sensor_config_sha'] == ds.config_sha and e['sensor_device'] == 'device1' and len(e['R_sha256']) == 64

def test_device2_dataset_builds(fake_sensor_root, ds):
    assert DS.HFDSensorSim(fake_sensor_root, 'test', _cfg(), device='device2', setting=dict(mode='sequential', snr_db=30.0), prior_path=ds.prior_path)[0]['y'].shape == (16, 16, 30)

@pytest.mark.dataset
def test_x64_matches_the_legacy_loader(data_root):
    from data_loader.HFD_dataset import HFD_data
    from data_loader.warmstart_paths import PRIOR
    leg = HFD_data(data_path=data_root, train_mode='image', eval_ratio=0.1, split='test', data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=1)
    new = DS.HFDSensorSim(data_root, 'test', dict(default_training_fields('p' * 64), P_y=50.0), setting=dict(mode='sequential', snr_db=60.0), prior_path=PRIOR)
    assert new.img_list == leg.img_list
    x_leg, y_leg = leg[3]; it = new[3]
    np.testing.assert_allclose(it['x'], x_leg, atol=1e-5)
    np.testing.assert_allclose(it['y'] - (y_leg + leg.sensor_R_matrix.sum(0)) / 2, 0, atol=6 * it['sigma'])   # physical y = (loader y + R^T 1) / 2 up to noise
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, module missing.

- [ ] **Step 3: Implement `data_loader/hfd_sensor_sim.py`** per Interfaces. `__getitem__`: x01 = load_cube01 → (mode, snr) = draw_setting → σ = sigma_from_snr(snr, P_y) → (y, mask) = measure(x01, R, mode, σ, rng) → (x̂₆₄, s₆₄) = mmse_maps(y, mask if snapshot else None, R, prior, σ, A) → x = ((2 x01 − 1) @ A.T).astype(float32). The prior check of `HFDResidualData` (prior R equals the loader's R) is **not** applied: the prior's μ, C are device-independent; only `mu`, `C`, `s` are read.

- [ ] **Step 4: Run to verify pass** (the dataset test runs here too). Expected: 8 passed.

- [ ] **Step 5: Commit** — `git add data_loader/hfd_sensor_sim.py tests/test_hfd_sensor_sim.py` ; `git commit -m "feat(sensor): HFDSensorSim — simulated EC measurement (family or fixed setting) with MMSE centre and posterior-std maps as the conditioning, per-epoch noise, dict collate (Phase 3 Task 5)"`.

