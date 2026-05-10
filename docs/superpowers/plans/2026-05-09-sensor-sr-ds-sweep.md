# Sensor-SR Downsample-Rate Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train independent sensor-SR diffusion models at `ds_sr ∈ {2, 3, 4, 6}` on 64×64 images, each saved to its own checkpoint folder for later side-by-side comparison.

**Architecture:** Two minimal code edits — add an `area_resize` downsample method to `HFD_SensorSR_data` (uses `F.interpolate(mode='area')`) so non-integer-divisible rates like 3 and 6 work on 64×64 inputs, and add it to the entrypoint's CLI choices. Plus one parametrized SLURM template + a 5-line shell launcher. The existing `ds=8` run is left untouched.

**Tech Stack:** PyTorch (existing), SLURM, bash. No new dependencies.

**Reference spec:** `docs/superpowers/specs/2026-05-09-sensor-sr-ds-sweep-design.md`

**Testing convention:** This repo has no pytest suite. Smoke tests go in `if __name__ == "__main__":` blocks and are executed via `python -m path.to.file`. Every Python invocation **must** be preceded by activating the `hsi` conda env (see Step 0 below).

---

## Step 0: Conda env activation (run once per shell)

Every `python` command in this plan assumes the `hsi` conda env is active. If you are not sure, run:

```bash
source /mnt/shared-scratch/Katehi_L/chaoyi_he/conda/miniconda/bin/activate
conda activate /mnt/shared-scratch/Katehi_L/chaoyi_he/conda_envs/hsi
which python  # should print .../conda_envs/hsi/bin/python
```

If `which python` shows the system Python, the env is not active and `python -m ...` will fail on `import torch`.

---

## File Structure

Modified:
- `data_loader/HFD_sensor_sr_dataset.py` — add `area_resize` branch in `_downsample`, relax assertion, extend `__main__` smoke.
- `main_2d_sensor_sr_fsdp.py` — add `'area_resize'` to `--sr_downsample_method` argparse choices.

New:
- `job_sensor_sr_sweep.slurm` — parametrized SLURM template (reads `DS_SR` from env).
- `submit_sweep.sh` — launches 4 `sbatch` jobs (ds ∈ {2, 3, 4, 6}).

Untouched (existing run kept intact):
- `job_sensor_sr.slurm` — still runs `ds=8` from prior config.
- `results/2d_sensor_sr/ds8_strided/...` — existing checkpoints stay.

---

## Task 1: Add `area_resize` downsample method to dataset

**Files:**
- Modify: `data_loader/HFD_sensor_sr_dataset.py`

This task adds a third downsample method that uses `F.interpolate(mode='area')`. Unlike `strided` and `avg_pool`, it works for arbitrary integer downsample rates (including those that don't evenly divide the image size).

- [ ] **Step 1.1: Extend the `__main__` smoke block to cover `area_resize` for rates {2, 3, 4, 6}**

This is the failing test. Adding it first means we can run it once before code changes and confirm it fails for the expected reason (assertion at line 152), then run it again after the code change to confirm it passes.

Locate the existing `__main__` block at the bottom of `data_loader/HFD_sensor_sr_dataset.py` (currently around line 176–209). After the `assert fb.shape[1] == 30 and lb.shape[1] == 30` line and before `print('OK')`, insert:

```python
    # --- area_resize sweep smoke ---
    # Validates the new method on rates that both divide and don't divide H=64.
    expected_lowres = {2: 32, 3: 21, 4: 16, 6: 11}
    for rate in (2, 3, 4, 6):
        ds_rate = HFD_SensorSR_data(
            data_path='dataset/HFD100 Mat dataset',
            stats_path='dataset/HFD100 Mat dataset/sensor_stats_R1.json',
            sr_downsample_rate=rate,
            sr_downsample_method='area_resize',
        )
        full_r, low_r = ds_rate[0]
        assert full_r.shape[0] == full_r.shape[1], \
            f'expected square full image, got {full_r.shape}'
        side = expected_lowres[rate]
        assert low_r.shape[0] == side and low_r.shape[1] == side, (
            f'ds={rate} area_resize: expected lowres {side}x{side}, '
            f'got {low_r.shape[0]}x{low_r.shape[1]}'
        )
        assert low_r.shape[-1] == 30, f'ds={rate}: lowres channels {low_r.shape[-1]} != 30'
        assert low_r.min() >= -1.0 - 1e-6 and low_r.max() <= 1.0 + 1e-6, \
            f'ds={rate}: lowres out of [-1,1]: [{low_r.min():.4f}, {low_r.max():.4f}]'
        print(f'  area_resize ds={rate}: full={full_r.shape} low={low_r.shape} '
              f'range=[{low_r.min():.4f}, {low_r.max():.4f}]')

    # area_resize and avg_pool must agree on integer-divisible rates.
    ds_area = HFD_SensorSR_data(
        data_path='dataset/HFD100 Mat dataset',
        stats_path='dataset/HFD100 Mat dataset/sensor_stats_R1.json',
        sr_downsample_rate=2, sr_downsample_method='area_resize',
    )
    ds_pool = HFD_SensorSR_data(
        data_path='dataset/HFD100 Mat dataset',
        stats_path='dataset/HFD100 Mat dataset/sensor_stats_R1.json',
        sr_downsample_rate=2, sr_downsample_method='avg_pool',
    )
    _, low_area = ds_area[0]
    _, low_pool = ds_pool[0]
    assert low_area.shape == low_pool.shape, \
        f'area vs pool shape mismatch: {low_area.shape} vs {low_pool.shape}'
    diff = float(np.abs(low_area - low_pool).max())
    assert diff < 1e-5, f'area_resize and avg_pool disagree on ds=2: max abs diff {diff:.2e}'
    print(f'  area_resize ≈ avg_pool on ds=2: max abs diff {diff:.2e}')
```

- [ ] **Step 1.2: Run the smoke to verify it fails**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
python -m data_loader.HFD_sensor_sr_dataset
```

Expected output: prints the existing `ds=8` smoke results, then crashes inside the new sweep loop. Either:
- `AssertionError: sr_downsample_method` (from the existing `assert sr_downsample_method in (...)` at line 36), **or**
- `AssertionError: image 64x64 not divisible by ds=3` (line 152) — depending on which guard fires first.

Either failure proves the new code path doesn't exist yet, which is what we want before implementing it.

- [ ] **Step 1.3: Add `'area_resize'` to the method whitelist**

Edit `data_loader/HFD_sensor_sr_dataset.py:36`:

```python
        assert sr_downsample_method in ('strided', 'avg_pool', 'area_resize')
```

- [ ] **Step 1.4: Add the `area_resize` branch to `_downsample`**

Edit `_downsample` (currently lines 124–137 in `data_loader/HFD_sensor_sr_dataset.py`). Replace the entire method body with:

```python
    def _downsample(self, full_sensor):
        ds = self.sr_downsample_rate
        if ds <= 1:
            return full_sensor
        if self.sr_downsample_method == 'strided':
            return full_sensor[::ds, ::ds, :]
        if self.sr_downsample_method == 'avg_pool':
            H, W, C = full_sensor.shape
            Hn, Wn = H // ds, W // ds
            return (
                full_sensor[:Hn * ds, :Wn * ds, :]
                .reshape(Hn, ds, Wn, ds, C)
                .mean(axis=(1, 3))
            )
        # area_resize: F.interpolate area mode — works for any integer rate,
        # including rates that don't evenly divide H/W. Output size is
        # round(H/ds) × round(W/ds), preserving signal energy via area weighting.
        H, W, C = full_sensor.shape
        Hn, Wn = int(round(H / ds)), int(round(W / ds))
        t = torch.from_numpy(full_sensor).permute(2, 0, 1).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(Hn, Wn), mode='area')
        return t.squeeze(0).permute(1, 2, 0).contiguous().numpy()
```

- [ ] **Step 1.5: Relax the divisibility assertion in `__getitem__`**

Edit `data_loader/HFD_sensor_sr_dataset.py:151–155`. Replace:

```python
        H, W, C = gt.shape
        assert H % self.sr_downsample_rate == 0 and \
               W % self.sr_downsample_rate == 0, (
            f'image {H}x{W} not divisible by ds={self.sr_downsample_rate}'
        )
```

with:

```python
        H, W, C = gt.shape
        if self.sr_downsample_method != 'area_resize':
            assert H % self.sr_downsample_rate == 0 and \
                   W % self.sr_downsample_rate == 0, (
                f'image {H}x{W} not divisible by ds={self.sr_downsample_rate}; '
                f'use sr_downsample_method=area_resize for non-divisible rates'
            )
```

- [ ] **Step 1.6: Run the smoke to verify it passes**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
python -m data_loader.HFD_sensor_sr_dataset
```

Expected output (last lines):

```
  area_resize ds=2: full=(64, 64, 30) low=(32, 32, 30) range=[...]
  area_resize ds=3: full=(64, 64, 30) low=(21, 21, 30) range=[...]
  area_resize ds=4: full=(64, 64, 30) low=(16, 16, 30) range=[...]
  area_resize ds=6: full=(64, 64, 30) low=(11, 11, 30) range=[...]
  area_resize ≈ avg_pool on ds=2: max abs diff <some small number>
OK
```

If any assertion fires, fix the implementation before proceeding. Common pitfalls:
- `torch.from_numpy` requires C-contiguous input — the `full_sensor` output from `_normalize` is already contiguous, but if you change the order of operations confirm with `.is_contiguous()`.
- `F.interpolate` with `mode='area'` expects 4D input `[N, C, H, W]` — the `unsqueeze(0)` after `permute(2, 0, 1)` covers that.
- The final `.contiguous()` matters because `permute` returns a non-contiguous view, and downstream `np.stack` in the collate fn relies on contiguous arrays.

- [ ] **Step 1.7: Commit**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
git add data_loader/HFD_sensor_sr_dataset.py
git commit -m "$(cat <<'EOF'
Add area_resize downsample method for non-divisible ds rates

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Expose `area_resize` to the FSDP entrypoint CLI

**Files:**
- Modify: `main_2d_sensor_sr_fsdp.py:62-63`

- [ ] **Step 2.1: Verify the current CLI rejects `area_resize`**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
python main_2d_sensor_sr_fsdp.py \
    --sensor_stats_path "dataset/HFD100 Mat dataset/sensor_stats_R1.json" \
    --save_path /tmp/throwaway \
    --sr_downsample_method area_resize \
    --help 2>&1 | tail -5
```

Expected: argparse error mentioning `invalid choice: 'area_resize'`. (The `--help` flag normally suppresses argument validation, but bad choices in `choices=[...]` are caught before help renders. If argparse instead prints help, no harm done — `area_resize` simply isn't in the choices yet, which Step 2.2 fixes.)

- [ ] **Step 2.2: Add `area_resize` to the choices**

Edit `main_2d_sensor_sr_fsdp.py:62-63`. Replace:

```python
    p.add_argument('--sr_downsample_method', type=str, default='strided',
                   choices=['strided', 'avg_pool'])
```

with:

```python
    p.add_argument('--sr_downsample_method', type=str, default='strided',
                   choices=['strided', 'avg_pool', 'area_resize'])
```

- [ ] **Step 2.3: Verify the CLI now accepts `area_resize`**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
python -c "
import argparse, sys
sys.argv = ['x', '--sensor_stats_path', '/tmp/x', '--save_path', '/tmp/x',
            '--sr_downsample_method', 'area_resize', '--ds_sr', '3']
from main_2d_sensor_sr_fsdp import get_parser
args = get_parser().parse_args()
print('parsed:', args.sr_downsample_method, args.ds_sr)
"
```

Expected: `parsed: area_resize 3`

- [ ] **Step 2.4: Commit**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
git add main_2d_sensor_sr_fsdp.py
git commit -m "$(cat <<'EOF'
Expose area_resize downsample method via CLI

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Create the parametrized SLURM template

**Files:**
- Create: `job_sensor_sr_sweep.slurm`

This is a near-copy of `job_sensor_sr.slurm` with three differences: it reads `DS_SR` from environment, names itself `ssr_ds${DS_SR}`, and writes to `ds${DS_SR}_area/...` instead of `ds8_strided/...`. It does NOT pass `--resume` (each run starts fresh per the spec).

- [ ] **Step 3.1: Create `job_sensor_sr_sweep.slurm`**

Write the following file at the repo root:

```bash
#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=ssr_ds_sweep
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH -o sensor_sr_sweep_%j.log
#SBATCH -e sensor_sr_sweep_%j.err

# DS_SR is required and must be passed via:
#   sbatch --export=ALL,DS_SR=<rate> job_sensor_sr_sweep.slurm
if [[ -z "$DS_SR" ]]; then
    echo "ERROR: DS_SR environment variable not set." >&2
    echo "Submit with: sbatch --export=ALL,DS_SR=<rate> job_sensor_sr_sweep.slurm" >&2
    exit 1
fi

cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu/
source /mnt/shared-scratch/Katehi_L/chaoyi_he/conda/miniconda/bin/activate
conda activate /mnt/shared-scratch/Katehi_L/chaoyi_he/conda_envs/hsi

export PYTORCH_ALLOC_CONF=expandable_segments:True

which python
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"

GPUS_PER_NODE=2

STATS_PATH="dataset/HFD100 Mat dataset/sensor_stats_R1.json"
SAVE_PATH="results/2d_sensor_sr/ds${DS_SR}_area/HFD/R_1/l1_loss/3dconv"

echo "=========================================="
echo "Sensor-SR sweep run: DS_SR=${DS_SR}"
echo "Save path: ${SAVE_PATH}"
echo "=========================================="

torchrun \
    --standalone \
    --nproc_per_node=$GPUS_PER_NODE \
    main_2d_sensor_sr_fsdp.py \
    --sensor_stats_path "$STATS_PATH" \
    --batch_size 16 \
    --lr 1e-4 \
    --base_channels 256 \
    --use_channel_3d_conv \
    --channel_num_filters 4 \
    --prediction_type v \
    --lrf 0.033 \
    --ds_sr "$DS_SR" \
    --sr_downsample_method area_resize \
    --save_path "$SAVE_PATH" \
    --num_epochs 200
```

Notes for the implementer:
- The SLURM `--job-name` line is fixed (`ssr_ds_sweep`) because SLURM directives are parsed before the script body runs, so `${DS_SR}` would not be substituted there. The submit launcher in Task 4 overrides the job name per-rate via `--job-name=ssr_ds${ds}` on the `sbatch` command line.
- Same reasoning applies to `-o`/`-e` log paths — they keep `%j` (job id) which SLURM substitutes natively.

- [ ] **Step 3.2: Commit**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
git add job_sensor_sr_sweep.slurm
git commit -m "$(cat <<'EOF'
Add parametrized SLURM template for sensor-SR ds sweep

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Create the sweep launcher

**Files:**
- Create: `submit_sweep.sh`

- [ ] **Step 4.1: Create `submit_sweep.sh`**

Write the following file at the repo root:

```bash
#!/bin/bash
# Submit one SLURM job per downsample rate for the sensor-SR sweep.
# Each job runs independently (no shared state, no warm-start) and writes
# to its own results/2d_sensor_sr/ds${ds}_area/... folder.
set -euo pipefail

cd "$(dirname "$0")"

for ds in 2 3 4 6; do
    sbatch \
        --job-name="ssr_ds${ds}" \
        --export=ALL,DS_SR=${ds} \
        job_sensor_sr_sweep.slurm
done
```

- [ ] **Step 4.2: Make it executable**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
chmod +x submit_sweep.sh
```

- [ ] **Step 4.3: Dry-run sanity check (does NOT submit)**

Verify the script parses cleanly and resolves paths without actually calling `sbatch`:

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
bash -n submit_sweep.sh && echo "syntax OK"
```

Expected: prints `syntax OK`.

- [ ] **Step 4.4: Commit**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
git add submit_sweep.sh
git commit -m "$(cat <<'EOF'
Add launcher for sensor-SR ds-rate sweep

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Final pre-submit verification

**Files:** none (verification only)

Before handing off to the user for `sbatch`, verify the full pipeline once more from a clean shell.

- [ ] **Step 5.1: Re-run the dataset smoke**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
source /mnt/shared-scratch/Katehi_L/chaoyi_he/conda/miniconda/bin/activate
conda activate /mnt/shared-scratch/Katehi_L/chaoyi_he/conda_envs/hsi
python -m data_loader.HFD_sensor_sr_dataset
```

Expected last line: `OK`. Any other output is a regression — investigate before proceeding.

- [ ] **Step 5.2: Confirm the four target save paths do not already exist**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
for ds in 2 3 4 6; do
    p="results/2d_sensor_sr/ds${ds}_area/HFD/R_1/l1_loss/3dconv"
    if [[ -e "$p" ]]; then
        echo "WARNING: $p already exists — would clobber existing data"
        ls -la "$p"
    else
        echo "OK: $p (does not exist yet)"
    fi
done
```

Expected: four `OK:` lines. If any `WARNING:` appears, surface it to the user and stop — do not submit jobs that would overwrite checkpoints.

- [ ] **Step 5.3: Confirm the existing `ds=8` checkpoints are still intact**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
ls -la results/2d_sensor_sr/ds8_strided/HFD/R_1/l1_loss/3dconv/ 2>&1 | head -20
```

Expected: directory listing showing the existing `checkpoint_epoch_*.pth`, `args_sensor_sr.json`, etc. (proves we haven't accidentally renamed or removed it).

- [ ] **Step 5.4: Hand off to the user**

Tell the user: implementation is complete, all smoke tests pass, no clobber risk. They can now submit the sweep with:

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
./submit_sweep.sh
```

…then monitor with `squeue -u $USER`.

Do **not** run `submit_sweep.sh` yourself — submitting GPU jobs is a user-authorized action with non-trivial cluster impact. Per the system instructions, hand-off is the correct stopping point for this plan.

---

## Spec coverage check

| Spec section | Covered by |
|---|---|
| §2 area_resize for non-divisible rates | Task 1 (Steps 1.3–1.5) |
| §3 sweep config: rates, init, epochs, method, stats | Task 3 (slurm content), Task 4 (rate list) |
| §4.1 dataset changes | Task 1 |
| §4.2 CLI changes | Task 2 |
| §4.3 parametrized SLURM | Task 3 |
| §4.4 launcher | Task 4 |
| §5 output layout (ds${ds}_area folders) | Task 3 (Step 3.1 SAVE_PATH), Task 5 (Step 5.2 clobber check) |
| §6.1 dataset CPU smoke | Task 1 (Steps 1.1–1.6), Task 5 (Step 5.1 re-run) |
| §6.2 CLI dry-run | Task 2 (Step 2.3) |
| §8 risks (cluster contention, area vs strided non-comparability) | Documented in spec; no implementation work needed |

No spec sections lack a task.
