### Task 12: Stage 0 script, the normalised-vs-raw check, and the CPU smoke

**Files:**
- Create: `scripts/sensor/stage0_baselines.py`, `scripts/pilot/smoke_sensor.sh`
- Test: `tests/test_stage0.py`

**Interfaces:**
- Consumes: `HFDSensorSim` (with a `sensor_override` hook: `HFDSensorSim.__init__(..., R_override: np.ndarray | None = None)` added here — when given, `self.R = R_override` and `R_sha256` is of it; add this one-line hook to Task 5's class), `mmse.demosaic_then_mmse`, `sensor_model.load_sensor`, `uq_eval.gaussian_metrics / point_metrics`, `scipy.linalg.subspace_angles`.
- Produces: `STAGE0_DIR = os.path.join(SENSOR_DIR, 'stage0')`; `SNRS = (60.0, 50.0, 40.0, 30.0, 20.0)`; `def run_L(ds, loader, device) -> dict(per_cube=dict(L=..., L_demosaic=... if snapshot), files)`; `def raw_sigma(R_raw, files, snr_db) -> float` (σ for the raw variant from the gate cubes' own mean y² under R_raw, so both variants sit at the same SNR); `def shape_check(curve_norm: dict[snr -> rmse], curve_raw) -> dict(monotone_norm, monotone_raw, max_dev, passed)` (both curves non-increasing in SNR; `ratios = raw / norm`; `max_dev = max |ratio − median(ratios)|`; passed iff both monotone and max_dev ≤ 0.10); `def principal_angles(R_a, R_b) -> list[float]` (degrees, `subspace_angles`); CLI `--data_root --prior --sensor_config --device cuda:0 --limit_files 0`: for device ∈ (device1, device2, ph5) × mode ∈ (sequential, snapshot) × SNRS: L (and L_demosaic) with the normalised R on the 200 gate cubes → `stage0/L__<device>__<mode>__<snr>.json`; the same with `R_raw` for device1 only → `stage0/Lraw__device1__<mode>__<snr>.json`; `stage0/summary.json` with the two curves per mode, `shape_check` results, the principal angles device1–device2 and device1–ph5, and the 60 dB sequential device1 L RMSE as `sanity_reference_rmse_pct`; prints `NORMALISATION CHECK: PASS|FAIL (max_dev=...)`.
- `scripts/pilot/smoke_sensor.sh`: CPU, `set -euo pipefail`, results root `${RESIDUAL_WARMSTART_RESULTS:-...}/smoke_sensor_<timestamp>` (removed at the end only if the script created it): `make_config.py --out $T/config.json --limit_files 16 --n_e2_files 4` → four trainings via `main_2d_residual_fsdp.py` directly (`--mode <m> --ds 1 --sensor_config $T/config.json --base_channels 16 --batch_size 4 --num_workers 2 --device cpu --num_epochs 1 --save_every 1 --val_every 1 --limit_files 16 --save_dir $T/runs/<m>` with the mode's prediction / loss flags) → `sd_eval.py` on sequential 30 dB and snapshot 30 dB with `--methods L L_demosaic D V C B --K 3 --pigdm_steps 2 --limit_files 8 --subset ddpm --sensor_config $T/config.json --prior ...` → `sd_verdict.py --manifest $T/manifest.json --report_only` → `toy_guidance.py --out $T/toy.json --steps 5 --K 8 --B 1 --hw 4 --device cpu` → prints `SMOKE OK`.

- [ ] **Step 1: Write the failing tests**

```python
def test_shape_check_passes_same_shape_and_fails_on_a_kink()      # raw = 1.3 * norm -> passed; raw with one SNR scaled by 1.2 -> max_dev > 0.10 -> failed; a non-monotone curve -> failed
def test_principal_angles_identity_and_orthogonal()                # angles(R, R) all ~0; angles of disjoint-support filters > 0
def test_raw_sigma_matches_nominal_snr(small_R)
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, module missing.

- [ ] **Step 3: Implement `scripts/sensor/stage0_baselines.py`, the `R_override` hook, and `scripts/pilot/smoke_sensor.sh`.**

- [ ] **Step 4: Run the tests, then the smoke.** Run: `... -m pytest tests/test_stage0.py tests/test_hfd_sensor_sim.py -q` → pass; `CUDA_VISIBLE_DEVICES= bash scripts/pilot/smoke_sensor.sh` → ends with `SMOKE OK` (≈ 10 min on CPU). Then the whole suite: `... -m pytest tests -q -x` → all pass (the Phase 1/2a suites included).

- [ ] **Step 5: Commit** — `git add scripts/sensor/stage0_baselines.py scripts/pilot/smoke_sensor.sh data_loader/hfd_sensor_sim.py tests/test_stage0.py` ; `git commit -m "feat(sensor): Stage 0 baselines with the normalised-vs-raw shape check and principal angles; CPU end-to-end smoke of the Phase 3 pipeline (Phase 3 Task 12)"`.

