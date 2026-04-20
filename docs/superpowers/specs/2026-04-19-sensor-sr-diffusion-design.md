# Sensor Super-Resolution Diffusion — Design Spec

Status: approved, ready for implementation planning
Date: 2026-04-19

## 1. Motivation

The existing HSI diffusion model (`U2NetHyperspectral` + `HFD_data`) maps a 30-channel sensor measurement to a 64-channel hyperspectral cube. It has plateaued around loss 0.35–0.36 on L1/v-prediction after ~20 epochs of the current 100-epoch schedule (`2d_diffusion_ds2_391145.log`).

This spec introduces a **separate** diffusion model that does not replace the existing one. It tackles a different subproblem: **sensor-domain spatial super-resolution**. Given a spatially downsampled sensor measurement, recover the full-resolution sensor measurement. Default downsample factor is 8.

Both models remain in the codebase; this one is additive.

## 2. Task & Tensor Contract

Let `R ∈ R^{31 × 30}` be the sensor response matrix (loaded from `R_Device{n}.mat`). For each ground-truth HSI image `gt ∈ R^{H × W × 31}`:

1. Normalize `gt` per-sample to `[-1, 1]` (same as current `HFD_data`).
2. Compute full-res sensor: `full_sensor = gt_norm @ R`, shape `[H, W, 30]`.
3. Normalize `full_sensor` to `[-1, 1]` using **global precomputed stats** (single scalar min/max per `(dataset, R_n)` pair). Clip to `[-1, 1]` for safety.
4. Build low-res conditioning via spatial downsample (`ds=8` default):
   - `strided`: `full_sensor[::ds, ::ds, :]`
   - `avg_pool`: `block_reduce(full_sensor, (ds, ds, 1), np.mean)`
5. Apply the **same** global normalization to the low-res tensor (consistency across x_0 and cond).

Diffusion inputs:

| symbol | shape | role |
|---|---|---|
| `x_0` | `[B, 30, H, W]` | target (full-res sensor) |
| `cond` | `[B, 30, H/ds, W/ds]` | conditioning (low-res sensor) |
| `t` | `[B]` | diffusion timestep |

The underlying `DiffusionTrainer` (`model/diffusion_trainer.py`) is task-agnostic and requires no changes — it calls `model(x_t, cond, t)` and supports eps/x0/v prediction types with L1/L2 loss.

## 3. Architecture

### 3.1 New model: `model/u2net_sensor_sr.py`

`U2NetSensorSR` mirrors `U2NetHyperspectral` but with two targeted changes:

- `in_channels = out_channels = sensor_channels` (both 30 by default).
- **Context encoder operates on low-res input.** The existing context encoder assumes context and x_0 share spatial resolution; here context is 8× smaller. The new encoder:
  1. Runs Conv3×3 / GN / GELU / Conv3×3 / GN / GELU / Conv1×1 on `[B, 30, H/ds, W/ds]` to produce `[B, base_C*8, H/ds, W/ds]`.
  2. Inside `forward`, for each U²-Net stage resolution, `F.interpolate(..., mode='bilinear', align_corners=True)` the context to `(H, H/2, H/4, H/8)` as needed, then reshape to `[B, hw, base_C*8]` and add time embedding. This preserves the cross-attention pattern used in `U2NetHyperspectral`.

Everything else — FiLM time conditioning, `U2NetBlock2D` nesting, `BasicTransformerBlock` attention, pooling / upsampling layout, skip concats, final projection — is unchanged.

### 3.2 3D-conv switchability

The `use_channel_3d_conv` toggle is plumbed through identically to `u2net_hyperspectral.py`:

- Constructor accepts `use_channel_3d_conv`, `channel_kernel`, `spatial_kernel`, `channel_num_filters`.
- When true, every `ConvBlock2D` inside `U2NetBlock2D` and the bridge becomes `ConvBlock2D_ChannelAware`.
- The context encoder's Conv2d layers are unaffected (they sit outside the backbone).
- CLI flag `--use_channel_3d_conv` is carried into the new entrypoint.

Both 2D and 3D conv backbones work without additional changes because the context encoder is decoupled from the conv-block implementation.

## 4. Dataset

### 4.1 New class: `data_loader/HFD_sensor_sr_dataset.py`

`HFD_SensorSR_data` is a new class (not a subclass) that reuses logic from `HFD_data`:

- `.mat` scan, train/test split, `sensor_R_matrix` loading (`load_sensor_response_new`).
- Same per-sample gt → [-1, 1] normalization.
- Same `gt @ R` sensor computation.

New behavior:

- Loads global normalization stats (JSON) at init. Missing file → hard fail with remediation message.
- `__getitem__` returns `(full_sensor, lowres_sensor)` as `[H, W, 30]` / `[H/ds, W/ds, 30]` float32 arrays.
- Asserts `H % ds == 0` and `W % ds == 0` at construction.
- A new `image_collate_fn` variant stacks and permutes to `[B, 30, H, W]` / `[B, 30, H/ds, W/ds]`.

`HFD_data` itself is **not modified** — existing HSI-diffusion training stays bit-exact reproducible.

### 4.2 New script: `scripts/compute_sensor_stats.py`

One-shot utility run per `(dataset, R_n)` pair:

- Instantiates `HFD_data(split='train', data_format='image', type='Flower', R_n=n, sensor_down_sample_rate=1)` and iterates it to read each image's `gt_data`.
- Computes `full_sensor = gt_norm @ R` for each sample.
- Collects global scalar min and max over all samples and all pixels/channels.
- Writes `dataset/HFD100 Mat dataset/sensor_stats_R{n}.json` with shape `{"sensor_min": float, "sensor_max": float, "n_samples": int, "dataset": str, "R_n": int}`.

Device-specific because `R_Device1.mat` and `R_Device2.mat` have different response curves → different sensor-value ranges.

## 5. Training Entrypoint

### 5.1 New files

- `main_2d_sensor_sr_fsdp.py` — adapted from `main_2d_fsdp.py`.
- `train_eval/train_sensor_sr.py` — adapted from `train_eval/train_2d.py`.
- `job_sensor_sr.slurm` — adapted from `job_fsdp.slurm`.

### 5.2 CLI changes relative to `main_2d_fsdp.py`

Added:

- `--ds_sr` int, default 8 — spatial downsample rate for conditioning.
- `--sr_downsample_method` str, default `strided`, choices `{strided, avg_pool}`.
- `--sensor_stats_path` str, required — path to JSON stats file.

Removed (not relevant to this task):

- `--spectral_channels` — no spectral output.
- `--sensor_down_sample_rate`, `--ds_warmup_epochs` — not relevant; the SR task always downsamples.

Kept unchanged:

- All FSDP args (`--sharding_strategy`, `--wrap_min_params`, `--cpu_offload`, `--no_mixed_precision`).
- All diffusion args (`--loss_type`, `--noise_schedule`, `--timesteps`, `--prediction_type`).
- All optimisation args (`--lr`, `--lrf`, `--num_epochs`, `--weight_decay`, `--grad_clip`).
- 3D-conv args (`--use_channel_3d_conv`, `--channel_kernel`, `--spatial_kernel`, `--channel_num_filters`).

### 5.3 Output layout

```
results/2d_sensor_sr/ds{ds_sr}_{method}/HFD/R_{n}/{conv_tag}/checkpoint_epoch_XX.pth
```

where `conv_tag ∈ {"2dconv", "3dconv"}` based on `--use_channel_3d_conv`.

## 6. Components Reused Unchanged

- `model/layers.py` — all conv blocks (`ConvBlock2D`, `ConvBlock2D_ChannelAware`), `BasicTransformerBlock`, `TimeEmbedding`.
- `model/diffusion_trainer.py` — task-agnostic; handles q_sample, get_loss, sample for eps/x0/v.
- `misc/util.py` — `MetricLogger`, `SmoothedValue`.
- FSDP wrapping, checkpoint save/load (`FullStateDictConfig`, `FullOptimStateDictConfig`) — copied verbatim from `main_2d_fsdp.py`.

## 7. Validation & Error Handling

- Stats file missing → fail at dataset init with a message pointing at `scripts/compute_sensor_stats.py`.
- `H % ds_sr != 0` → assertion error at dataset init.
- Clip normalized sensor values to `[-1, 1]` after global scaling to handle outlier samples.
- Inference path (sampling) applies the same saved global stats — no train/inference gap.

## 8. Out of Scope

- Modifying the existing HSI-diffusion model (`U2NetHyperspectral`) or its dataset (`HFD_data`).
- Multi-scale / curriculum downsampling (rejected during brainstorming: single `ds_sr` per run).
- Learnable normalization / VAE latent space for sensor data.
- Multi-device stats fusion (stats are per `R_n`, not merged).

## 9. File Manifest

New files:

- `model/u2net_sensor_sr.py`
- `data_loader/HFD_sensor_sr_dataset.py`
- `scripts/compute_sensor_stats.py`
- `main_2d_sensor_sr_fsdp.py`
- `train_eval/train_sensor_sr.py`
- `job_sensor_sr.slurm`

Modified files: none.

Generated artifacts (checked into dataset directory, not git):

- `dataset/HFD100 Mat dataset/sensor_stats_R{n}.json`
