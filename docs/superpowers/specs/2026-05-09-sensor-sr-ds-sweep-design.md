# Sensor-SR Downsample-Rate Sweep — Design Spec

Status: approved, ready for implementation planning
Date: 2026-05-09

## 1. Motivation

The sensor-SR diffusion (`U2NetSensorSR` + `HFD_SensorSR_data`) currently has one in-progress training run at `ds_sr=8`. To characterize how reconstruction quality scales with downsample factor, we want to train independent models at `ds_sr ∈ {2, 3, 4, 6}` and save each to its own checkpoint folder for later side-by-side evaluation.

The existing `ds_sr=8` run is left untouched.

## 2. Constraint: 64×64 image size

HFD100 ground-truth images in this codebase are 64×64. The dataset's current divisibility assertion (`HFD_sensor_sr_dataset.py:152`) blocks `ds=3` and `ds=6` because:

- 64 / 2 = 32 ✓
- 64 / 3 = 21.33 ✗
- 64 / 4 = 16 ✓
- 64 / 6 = 10.67 ✗

To support arbitrary integer downsample rates, this spec adds a third `sr_downsample_method` — **`area_resize`** — that uses `torch.nn.functional.interpolate(..., mode='area')` to produce a low-res tensor of size `(round(H/ds), round(W/ds))`. For integer-divisible cases, `area_resize` is numerically equivalent to `avg_pool`, so using it for all four sweep rates yields a clean apples-to-apples comparison.

Resulting low-res shapes for 64×64:

| ds | low-res |
|---|---|
| 2 | 32×32 |
| 3 | 21×21 |
| 4 | 16×16 |
| 6 | 11×11 |

The `U2NetBlock2D.downsample_context` square-image assumption (`int(math.sqrt(hw))`) holds since each low-res output is square: `sqrt(N²) = N` exactly.

## 3. Sweep configuration

| key | value |
|---|---|
| rates | `{2, 3, 4, 6}` |
| init | from scratch, independent (no warm-start, no curriculum) |
| epochs | 200 (matches existing `ds=8` run) |
| downsample method | `area_resize` (consistent across all four) |
| stats file | reuse `dataset/HFD100 Mat dataset/sensor_stats_R1.json` (rate-independent) |
| scheduling | 4 separate `sbatch` jobs, parallel if the queue allows |

All other CLI args (batch size, lr, base channels, prediction type, loss type, etc.) match the existing `ds=8` run defaults from `job_sensor_sr.slurm`.

## 4. Code changes

### 4.1 `data_loader/HFD_sensor_sr_dataset.py`

- Extend the `sr_downsample_method` choices to include `'area_resize'` (line 36).
- Add a third branch to `_downsample`: convert the `[H, W, C]` numpy array to a `[1, C, H, W]` torch tensor, run `F.interpolate(size=(round(H/ds), round(W/ds)), mode='area')`, convert back to numpy.
- Relax the divisibility assertion at line 152 to apply only to `strided` and `avg_pool`. Update the failure message to point users to `area_resize` for non-divisible rates.

### 4.2 `main_2d_sensor_sr_fsdp.py`

- Add `'area_resize'` to the `--sr_downsample_method` CLI choices (line 62-63). No other changes — `--ds_sr` already accepts arbitrary integers.

### 4.3 New: `job_sensor_sr_sweep.slurm`

Parametrized SLURM template that reads `DS_SR` from environment. Existing `job_sensor_sr.slurm` is left in place for the `ds=8` run.

Key differences from the existing job script:

- `--job-name=ssr_ds${DS_SR}` so `squeue` distinguishes the four jobs.
- `--save_path=results/2d_sensor_sr/ds${DS_SR}_area/HFD/R_1/l1_loss/3dconv` (note `_area` suffix to distinguish from the existing `ds8_strided` folder).
- `--sr_downsample_method area_resize`
- `--ds_sr ${DS_SR}`
- No `--resume` flag (each run starts fresh).
- Fail-loud check at the top: `[[ -z "$DS_SR" ]] && { echo "DS_SR not set"; exit 1; }`.

### 4.4 New: `submit_sweep.sh`

Tiny launcher:

```bash
#!/bin/bash
for ds in 2 3 4 6; do
    sbatch --export=ALL,DS_SR=$ds job_sensor_sr_sweep.slurm
done
```

## 5. Output layout

```
results/2d_sensor_sr/
├── ds8_strided/HFD/R_1/l1_loss/3dconv/      # existing run (untouched)
├── ds2_area/HFD/R_1/l1_loss/3dconv/         # new
├── ds3_area/HFD/R_1/l1_loss/3dconv/         # new
├── ds4_area/HFD/R_1/l1_loss/3dconv/         # new
└── ds6_area/HFD/R_1/l1_loss/3dconv/         # new
```

Each new folder receives the same artifacts as the existing run: `args_sensor_sr.json`, `checkpoint_epoch_*.pth`, `best_model.pth`, `final_model.pth`, `train_history.{txt,json}`.

## 6. Verification

Before submitting the four SLURM jobs:

1. **Dataset CPU smoke** — run the `__main__` block in `HFD_sensor_sr_dataset.py` (or a small ad-hoc script) for each rate ∈ {2, 3, 4, 6} with `area_resize`. Verify:
   - `lowres.shape[1:3] == (round(64/ds), round(64/ds))`
   - `lowres.min() >= -1.0 - 1e-6` and `lowres.max() <= 1.0 + 1e-6`
   - No exceptions.

2. **CLI dry-run** — confirm `--sr_downsample_method area_resize` is accepted by the entrypoint argparser.

If both pass, submit via `bash submit_sweep.sh`.

## 7. Out of scope

- Cross-rate evaluation / comparison script — defer until checkpoints exist.
- The known FSDP weak spots flagged in the prior memory (parent `main_2d_fsdp.py` final-save deadlock, `--resume` silent skip) — out of scope for this sweep, no behavior change here.
- L2 / Huber loss path — no change to loss configuration.

## 8. Risks

- **Cluster contention**: 4 simultaneous `gpu:a100:2` jobs may not all start at once. Expected; jobs will queue and that's fine.
- **`area_resize` ≠ `strided`**: results from this sweep are not directly comparable to the existing `ds=8` strided run. If a clean comparison against `ds=8` is needed later, retrain `ds=8` with `area_resize` (out of scope here).
