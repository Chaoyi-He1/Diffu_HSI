# Regressor-Centred Residual Diffusion for Calibrated Uncertainty (Phase 2a) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the cross-fitted, regressor-centred residual diffusion with its variance-head and ensemble baselines and the uncertainty / texture evaluation harness, so that the pre-registered d = 8 verdict of spec §5.7 can be taken once, with a d = 4 contrast.

**Architecture:** The centre of the diffusion is the one-call regressor's estimate x̂_R, cached per training cube from the half-data regressor that did not see it (cross-fitting) and, for monitoring and evaluation, from the full selected regressor. The cached centre travels in the existing `x0_hat` condition slot of `ResidualWrapper`, so the trainer's loops, the resume/meta machinery and the sampler are reused unchanged; only the residual scale (σ_R instead of σ_d) and two new modes (`residual_reg`, `varhead`) are added. A numpy metrics module (CRPS, coverage, AUSE, high-frequency ratio, gradient W1) feeds one evaluation script that writes one record per run and one verdict script that applies the pre-registered thresholds.

**Tech Stack:** Python 3.10, PyTorch (conda env `hsi`, `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python`), numpy, pytest; the existing `main_2d_residual_fsdp.py` trainer, `model/residual_wrapper.py`, `model/diffusion_trainer.py`, `scripts/eval_warmstart.py`, `scripts/paired_stats.py`, `scripts/pilot/launch_run.sh`.

**Spec:** `docs/superpowers/specs/2026-10-22-regressor-centred-uq-diffusion-design.md` (approved by the user 2026-10-22, commit 162c724). The B-track record it builds on: `docs/superpowers/plans/2026-10-19-b-track-prereg.md`.

## Global Constraints

- No noise is added to the sensor data; `data_loader/HFD_dataset.py` is never modified; `data_loader/hfd_residual.py` only gains the one-line `raw_item` helper of Task 3.
- Never write under `results/residual_warmstart/{eval,diagnostics}` or into an existing run directory of an earlier phase; new results go under `results/residual_warmstart/uq/` and new run directories under `results/residual_warmstart/runs/`.
- Every record-writing script refuses to overwrite an existing output file.
- Metrics are reported on the [0, 1] display scale in % (as `scripts/eval_warmstart.py::metrics`); residual scales σ_d and σ_R live on the trainer's [−1, 1] scale.
- σ_R is the RMS of all elements of x − x̂_R over the 9 000 cross-fitted training cubes, computed exactly as `scripts/compute_residual_scale.py` computes σ_d (RMS, float64 accumulation).
- Pre-registered centres: d = 8 → `runs/regress_l1_sel_d8_b64/checkpoint_epoch_11.pth`; d = 4 → `runs/regress_l1_sel_d4_b64/checkpoint_epoch_17.pth`. K = 32 samples per cube at d = 8, K = 16 at d = 4, sampler DDIM-10 from pure noise, seeds 0 … K−1.
- Thresholds (spec §5.7) are fixed; `scripts/uq_verdict.py` is the only place they are coded and it is written (Task 9) before any Stage 1 number exists.
- GPU runs go through `scripts/pilot/launch_run.sh`; never share a GPU with the user's `dm_snn` process (the launcher refuses a busy GPU).
- Tests run on CPU: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/<file> -q`. Tests that need the dataset carry `@pytest.mark.dataset`.
- Commit after every task with the trailer lines `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP`; never stage `model/__pycache__/*.pyc`; push with `git push git@github.com:Chaoyi-He1/Diffu_HSI.git worktree-phase1-plan`.
- The worktree hook refuses compound shell commands that mention git and heredocs: write helper scripts to files first, run them with plain commands.

## Design decisions (rulings that refine the spec; recorded in the spec's as-built note in Task 10)

- **D1 — the centre travels in the `x0_hat` slot.** For modes `residual_reg` and `varhead` the dataset wrapper `XRCachedData` returns `(x, y_d, x_r)` in the place of `(x, y_d, x0_hat)` and exposes `sigma_d = σ_R`, so `train_one_epoch`, `validate`, `target_for` and `reconstruct` run unchanged. The spec's separate `x_r` cond key is not added; the checkpoint meta carries explicit `sigma_r`, `centre` and `xr_cache_sha256` fields so nothing is ambiguous in the records.
- **D2 — `varhead` takes `--loss_type nll`.** `--loss_type` gains the choice `nll`; mode `varhead` requires it and every other mode refuses it (the spec's "`--loss_type` is refused" cannot be detected with argparse defaults; an explicit required value is stricter).
- **D3 — the cache's `val/` part covers all 1 000 test-split files** (monitoring + gate) with the full regressor, but `uq_eval.py` computes x̂_R for the gate cubes on the fly in float32 from the `--centre` checkpoint; the cache's val part serves only the trainer's monitoring loss. σ_full is computed on the 800 monitoring files only.
- **D4 — `--diffusion` and `--varhead` are optional in `uq_eval.py`** so the Stage 0 dry run (spec §7) can score the d = 4 assets before any new model exists.

## Review Focus

1. A cache built for one stride used with a dataset of another (`--ds 8` with a d = 4 cache): `XRCachedData` must refuse at construction — pinned in Task 3.
2. A half regressor that saw the cube (the builder handed the wrong half's checkpoint): `build_xr_cache.py` must refuse a `--half1` checkpoint whose meta `train_half` is not 1 (and `--half2` ≠ 2, `--full` ≠ 0) — pinned in Task 5.
3. Resuming a `residual_reg` run after the cache was rebuilt (different σ_R or files): the resume must be refused by the cache sha — pinned in Task 4.
4. Constant or near-constant samples (a collapsed sampler): `crps_samples` must refuse K < 2 and `coverage_samples` must count x == lo == hi as covered — pinned in Task 7.
5. A verdict taken on a record whose centre is not the pre-registered checkpoint of that stride: `uq_verdict.py` must refuse — pinned in Task 9.

## File structure

| path | responsibility |
|---|---|
| `scripts/uq/__init__.py` | package marker |
| `scripts/uq/split_halves.py` | the cross-fitting split by source image; `split_by_source_image`, `write_split`, CLI |
| `scripts/uq/select_epoch.py` | the pre-registered selection rule (lowest monitoring loss, ties → earlier) as a function + CLI |
| `model/residual_wrapper.py` | `MODES` + `XR_MODES`, `build_residual_model`, `target_for`, `varhead_forward`, `varhead_nll`, `compose` |
| `model/varhead_trainer.py` | `VarheadTrainer.get_loss` (Gaussian NLL), duck-typed like `RegressionTrainer` |
| `data_loader/xr_cache.py` | cache layout, meta read/write/sha, `XRCachedData` |
| `data_loader/hfd_residual.py` | `HFDResidualData.raw_item` (the base item without x̂₀) |
| `data_loader/warmstart_paths.py` | `UQ_DIR`, `SPLIT_JSON` |
| `main_2d_residual_fsdp.py` | `--train_half`, `--xr_cache`, `--split_json`, mode checks, dataset wrapping, meta fields, resume check |
| `scripts/uq/build_xr_cache.py` | writes the cache and its meta (σ_R, σ_full) |
| `scripts/pilot/launch_run.sh` | `_half<N>_` guard, `varhead` needs `--loss_type nll` |
| `scripts/uq/metrics.py` | numpy metrics: `crps_samples`, `crps_gaussian`, `coverage_samples`, `coverage_gaussian`, `ause`, `spread_ratio`, `hf_energy_ratio`, `grad_w1` |
| `scripts/uq_eval.py` | samples, scores every method, writes one record |
| `scripts/uq_verdict.py` | the pre-registered thresholds |
| `scripts/pilot/smoke_uq.sh` | CPU end-to-end smoke on 16 files |
| `tests/test_uq_split.py`, `tests/test_uq_modes.py`, `tests/test_xr_cache.py`, `tests/test_uq_training.py`, `tests/test_build_xr_cache.py`, `tests/test_launch_run_name.py` (extended), `tests/test_uq_metrics.py`, `tests/test_uq_eval.py`, `tests/test_uq_verdict.py` | tests |

---

### Task 1: Cross-fitting split and the selection rule

**Files:**
- Create: `scripts/uq/__init__.py` (empty), `scripts/uq/split_halves.py`, `scripts/uq/select_epoch.py`
- Modify: `data_loader/warmstart_paths.py` (add two constants after `SUBSET`)
- Test: `tests/test_uq_split.py`

**Interfaces:**
- Consumes: `scripts.paired_stats.source_image(path) -> 'P0xx/<img>'`; `data_loader.warmstart_paths.RESULTS_ROOT`, `DATA_ROOT`, `PRIOR`, `git_head`.
- Produces: `split_by_source_image(files: list[str], seed: int = 0) -> dict` with keys `seed`, `half1`, `half2` (sorted lists of the given file strings), `images_per_class` ({class: {'half1': n, 'half2': n}}); `write_split(path, split) -> None` (atomic, refuses an existing path); `read_split(path) -> dict`; `select_epoch(run_dir) -> dict(epoch, val_loss, last_epoch, last_val_loss, n_val)`; constants `UQ_DIR = RESULTS_ROOT/uq`, `SPLIT_JSON = UQ_DIR/split_halves.json`.

- [ ] **Step 1: Add the path constants**

In `data_loader/warmstart_paths.py`, directly after the line `SUBSET = os.path.join(REPO, 'tests', 'data', 'val_subset_200.json')` add:

```python
UQ_DIR = os.path.join(RESULTS_ROOT, 'uq')                     # Phase 2a results (spec §6)
SPLIT_JSON = os.path.join(UQ_DIR, 'split_halves.json')         # the cross-fitting split (spec §4.1)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_uq_split.py`:

```python
"""Phase 2a Task 1: the cross-fitting split keeps every patch of a source image in one half, balances the halves
within one image per class, is deterministic, and the selection rule picks the lowest monitoring loss."""
import json
import os

import pytest

from scripts.paired_stats import source_image
from scripts.uq.select_epoch import select_epoch
from scripts.uq.split_halves import read_split, split_by_source_image, write_split


def fake_files():
    """3 classes; class c has 5 + c images with 1-4 patches each (so halves cannot be balanced by patch count)."""
    files = []
    for c in range(3):
        for img in range(5 + c):
            for patch in range(1 + (img % 4)):
                files.append(f'MatFlower60/Train/P00{c + 1}/{img}_{patch}.mat')
    return files


def test_every_file_lands_in_exactly_one_half():
    files = fake_files()
    s = split_by_source_image(files, seed=0)
    assert sorted(s['half1'] + s['half2']) == sorted(files)
    assert not set(s['half1']) & set(s['half2'])


def test_patches_of_one_image_never_straddle_the_halves():
    s = split_by_source_image(fake_files(), seed=0)
    imgs1 = {source_image(f) for f in s['half1']}
    imgs2 = {source_image(f) for f in s['half2']}
    assert not imgs1 & imgs2


def test_halves_are_balanced_per_class_within_one_image():
    s = split_by_source_image(fake_files(), seed=0)
    for cls, n in s['images_per_class'].items():
        assert abs(n['half1'] - n['half2']) <= 1, (cls, n)
    assert set(s['images_per_class']) == {'P001', 'P002', 'P003'}
    assert s['images_per_class']['P003'] == {'half1': 4, 'half2': 3} or s['images_per_class']['P003'] == {'half1': 3, 'half2': 4}


def test_split_is_deterministic_and_depends_on_the_seed():
    a, b, c = (split_by_source_image(fake_files(), seed=s) for s in (0, 0, 1))
    assert a == b and a['half1'] != c['half1']
    assert a['seed'] == 0 and c['seed'] == 1


def test_halves_are_sorted_lists():
    s = split_by_source_image(fake_files(), seed=0)
    assert s['half1'] == sorted(s['half1']) and s['half2'] == sorted(s['half2'])


def test_write_split_round_trips_and_refuses_overwrite(tmp_path):
    s = split_by_source_image(fake_files(), seed=0)
    p = tmp_path / 'uq' / 'split_halves.json'
    write_split(str(p), s)
    assert read_split(str(p)) == s
    with pytest.raises(FileExistsError):
        write_split(str(p), s)
    assert not list(tmp_path.glob('**/*.tmp'))


def test_select_epoch_takes_the_lowest_monitoring_loss_ties_earlier(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    recs = [dict(epoch=1, loss=0.5, val_loss=0.45), dict(epoch=2, loss=0.4, val_loss=0.40),
            dict(epoch=3, loss=0.3, val_loss=0.40), dict(epoch=4, loss=0.2, val_loss=0.42)]
    (run / 'train_log.jsonl').write_text('\n'.join(json.dumps(r) for r in recs) + '\n')
    sel = select_epoch(str(run))
    assert sel == dict(epoch=2, val_loss=0.40, last_epoch=4, last_val_loss=0.42, n_val=4)


def test_select_epoch_skips_epochs_without_a_monitoring_loss(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    recs = [dict(epoch=1, loss=0.5), dict(epoch=2, loss=0.4, val_loss=0.41), dict(epoch=3, loss=0.3)]
    (run / 'train_log.jsonl').write_text('\n'.join(json.dumps(r) for r in recs) + '\n')
    assert select_epoch(str(run))['epoch'] == 2
    (run / 'train_log.jsonl').write_text(json.dumps(dict(epoch=1, loss=0.5)) + '\n')
    with pytest.raises(ValueError, match='no monitoring'):
        select_epoch(str(run))
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_uq_split.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'scripts.uq'`.

- [ ] **Step 4: Implement the split and the selector**

`scripts/uq/__init__.py`: empty file.

`scripts/uq/split_halves.py`:

```python
"""Cross-fitting split of the training cubes by SOURCE IMAGE (spec §4.1).

Source images (class/<img>, scripts.paired_stats.source_image) are grouped by class, shuffled with `seed` inside each
class and dealt alternately to half1 and half2, so no image has patches in both halves and the halves of a class
differ by at most one image. One split serves every stride. CLI (from the repo root):

    python scripts/uq/split_halves.py [--seed 0] [--out $RESIDUAL_WARMSTART_RESULTS/uq/split_halves.json]

reads the training file list exactly as the trainer does (HFDResidualData split='train' at d = 4; the file list does
not depend on d) and writes relative paths (relative to --data_root). Refuses to overwrite.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from data_loader.warmstart_paths import DATA_ROOT, PRIOR, SPLIT_JSON, git_head  # noqa: E402
from scripts.paired_stats import source_image  # noqa: E402


def split_by_source_image(files, seed=0):
    by_class = {}
    for f in files:
        key = source_image(f)
        by_class.setdefault(key.split('/')[0], {}).setdefault(key, []).append(f)
    rng = np.random.default_rng(seed)
    half1, half2, per_class = [], [], {}
    for cls in sorted(by_class):
        images = sorted(by_class[cls])
        order = rng.permutation(len(images))
        h1 = [images[i] for i in order[0::2]]
        h2 = [images[i] for i in order[1::2]]
        per_class[cls] = dict(half1=len(h1), half2=len(h2))
        for img in h1:
            half1 += by_class[cls][img]
        for img in h2:
            half2 += by_class[cls][img]
    return dict(seed=int(seed), half1=sorted(half1), half2=sorted(half2), images_per_class=per_class)


def write_split(path, split):
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; the split is written once and never overwritten')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(split, f, indent=1)
    os.replace(tmp, path)


def read_split(path):
    with open(path) as f:
        return json.load(f)


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=SPLIT_JSON)
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--prior', default=PRIOR)
    return ap


def main():
    args = get_parser().parse_args()
    from data_loader.hfd_residual import HFDResidualData
    ds = HFDResidualData(data_path=args.data_root, split='train', sensor_down_sample_rate=4, prior_path=args.prior)
    files = [os.path.relpath(f, args.data_root) for f in ds.img_list]
    split = split_by_source_image(files, args.seed)
    split.update(n_files=len(files), data_root=args.data_root, git_head=git_head())
    write_split(args.out, split)
    print(f"wrote {args.out}: {len(split['half1'])} + {len(split['half2'])} files, images per class {split['images_per_class']}")


if __name__ == '__main__':
    main()
```

`scripts/uq/select_epoch.py`:

```python
"""The pre-registered checkpoint-selection rule (B.1, spec §4.2 / §4.5): the epoch with the lowest monitoring loss in
train_log.jsonl, ties -> the earlier epoch. Usage: python scripts/uq/select_epoch.py RUN_DIR"""
import json
import os
import sys


def select_epoch(run_dir):
    with open(os.path.join(run_dir, 'train_log.jsonl')) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    vals = [(r['epoch'], float(r['val_loss'])) for r in recs if r.get('val_loss') is not None]
    if not vals:
        raise ValueError(f'{run_dir}: no monitoring loss in train_log.jsonl')
    epoch, val = min(vals, key=lambda t: (t[1], t[0]))
    return dict(epoch=int(epoch), val_loss=val, last_epoch=int(vals[-1][0]), last_val_loss=vals[-1][1], n_val=len(vals))


if __name__ == '__main__':
    print(json.dumps(select_epoch(sys.argv[1])))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_uq_split.py -q`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/uq/__init__.py scripts/uq/split_halves.py scripts/uq/select_epoch.py data_loader/warmstart_paths.py tests/test_uq_split.py
git commit -F <message file>   # "feat(uq): cross-fitting split by source image and the selection rule (Phase 2a Task 1)" + trailers
```

---

### Task 2: Modes `residual_reg` and `varhead`

**Files:**
- Modify: `model/residual_wrapper.py`
- Create: `model/varhead_trainer.py`
- Test: `tests/test_uq_modes.py`

**Interfaces:**
- Consumes: `ResidualWrapper`, `U2NetHyperspectral(extra_in_channels=...)`, `REGRESS_T`, `RegressionTrainer` (as the duck-type to copy).
- Produces: `MODES = ('residual', 'standard', 'standard_xhat', 'regress', 'residual_reg', 'varhead')`; `XR_MODES = ('residual_reg', 'varhead')`; `CENTRED_MODES = ('residual', 'regress', 'residual_reg', 'varhead')` (modes whose target is (x − centre)/σ); `varhead_forward(model, cond) -> logvar [B, 64, H, W]`; `LOGVAR_RANGE = (-10.0, 10.0)`; `varhead_nll(logvar, r) -> per-element tensor`; `compose(centre, sigma, r) -> clamp(centre + sigma r, -1, 1)`; `VarheadTrainer(device=..., n_timesteps=1000)` with `.get_loss(model, x_0, cond=None, t=None, *, loss_in_fp32=False) -> (loss, info)`, `.prediction_type == 'direct'`, `.loss_type == 'nll'`, `.last_nfe == 1`.

- [ ] **Step 1: Write the failing tests**

`tests/test_uq_modes.py`:

```python
"""Phase 2a Task 2: the two new modes share B's architecture (128 input channels); residual_reg has B's target with
the cached centre; varhead's one call sits on x_r at t = REGRESS_T and returns a log-variance scored by the Gaussian NLL."""
import math

import pytest
import torch

from model.residual_wrapper import (CENTRED_MODES, LOGVAR_RANGE, MODES, REGRESS_T, XR_MODES, build_residual_model,
                                    compose, target_for, varhead_forward, varhead_nll)
from model.varhead_trainer import VarheadTrainer


def test_mode_tables():
    assert MODES == ('residual', 'standard', 'standard_xhat', 'regress', 'residual_reg', 'varhead')
    assert XR_MODES == ('residual_reg', 'varhead') and set(XR_MODES) < set(MODES)
    assert CENTRED_MODES == ('residual', 'regress', 'residual_reg', 'varhead')


@pytest.mark.parametrize('mode', ['residual_reg', 'varhead'])
def test_new_modes_take_128_input_channels_like_b(mode):
    m = build_residual_model(mode, base_channels=8)
    assert m.use_x0_hat and m.net.input_proj.in_channels == 128
    x_t, xr, y = torch.zeros(2, 64, 16, 16), torch.rand(2, 64, 16, 16) * 2 - 1, torch.rand(2, 30, 4, 4)
    out = m(x_t, {'sensor': y, 'x0_hat': xr}, torch.full((2,), 10, dtype=torch.long))
    assert out.shape == (2, 64, 16, 16)


def test_residual_reg_and_varhead_targets_are_the_scaled_residual_to_the_centre():
    x, xr = torch.rand(2, 64, 4, 4) * 2 - 1, torch.rand(2, 64, 4, 4) * 2 - 1
    for mode in ('residual_reg', 'varhead'):
        assert torch.equal(target_for(mode, x, xr, 0.09), (x - xr) / 0.09)
    assert torch.equal(target_for('standard', x, xr, 0.09), x)


def test_varhead_forward_puts_the_centre_in_the_x_t_slot_at_regress_t():
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, x_t, cond, t):
            seen['x_t'], seen['t'] = x_t, t
            return torch.zeros_like(x_t)

    xr = torch.rand(3, 64, 8, 8)
    varhead_forward(Spy(), {'sensor': torch.zeros(3, 30, 2, 2), 'x0_hat': xr})
    assert torch.equal(seen['x_t'], xr) and torch.equal(seen['t'], torch.full((3,), REGRESS_T, dtype=torch.long))


def test_varhead_nll_closed_form_and_minimum():
    r = torch.tensor([2.0])
    assert torch.allclose(varhead_nll(torch.tensor([0.0]), r), torch.tensor([2.0]))          # 0.5 * (0 + 4)
    best = math.log(4.0)
    at_best = varhead_nll(torch.tensor([best]), r)
    assert at_best < varhead_nll(torch.tensor([best - 0.5]), r) and at_best < varhead_nll(torch.tensor([best + 0.5]), r)
    assert torch.allclose(at_best, torch.tensor([0.5 * (best + 1.0)]))


def test_varhead_nll_clamps_the_log_variance():
    r = torch.tensor([1.0])
    assert torch.equal(varhead_nll(torch.tensor([-50.0]), r), varhead_nll(torch.tensor([LOGVAR_RANGE[0]]), r))
    assert torch.equal(varhead_nll(torch.tensor([50.0]), r), varhead_nll(torch.tensor([LOGVAR_RANGE[1]]), r))


def test_compose_adds_the_scaled_residual_and_clamps():
    c = torch.tensor([0.9, -0.5]); r = torch.tensor([3.0, 1.0])
    assert torch.allclose(compose(c, 0.1, r), torch.tensor([1.0, -0.4]))


def test_varhead_trainer_loss_is_the_mean_nll_of_the_networks_logvar():
    class Const(torch.nn.Module):
        def forward(self, x_t, cond, t):
            return torch.full_like(x_t, 0.5)

    tr = VarheadTrainer(device=torch.device('cpu'))
    assert tr.prediction_type == 'direct' and tr.loss_type == 'nll' and tr.last_nfe == 1
    r = torch.full((2, 64, 4, 4), 1.5)
    loss, info = tr.get_loss(Const(), r, {'sensor': torch.zeros(2, 30, 1, 1), 'x0_hat': torch.zeros(2, 64, 4, 4)}, loss_in_fp32=True)
    expected = 0.5 * (0.5 + 1.5 ** 2 * math.exp(-0.5))
    assert abs(loss.item() - expected) < 1e-6 and info['t'] == REGRESS_T and info['logvar'].shape == r.shape
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_uq_modes.py -q`
Expected: ImportError on `CENTRED_MODES` / `model.varhead_trainer`.

- [ ] **Step 3: Implement**

In `model/residual_wrapper.py` replace the `MODES` block and `build_residual_model` / `target_for` and append the helpers, so the file's tail reads:

```python
MODES = ('residual', 'standard', 'standard_xhat', 'regress', 'residual_reg', 'varhead')
# residual      (B):  input [x_t, x0_hat], target (x - x0_hat) / sigma_d
# standard      (A):  input  x_t,          target  x
# standard_xhat (A'): input [x_t, x0_hat], target  x   -- B's inputs with A's target: controls for the spatially
#                     aligned x0_hat channels that the position-blind cross-attention context cannot provide
# regress       (R):  input [0, x0_hat] at t = REGRESS_T, target (x - x0_hat) / sigma_d predicted in ONE deterministic
#                     call (no noise, no sampler): B's architecture in B's one-step regime, trained as a regressor
# residual_reg  (RR, Phase 2a): B with the regressor's cached estimate x_r in the x0_hat slot (data_loader.xr_cache):
#                     input [x_t, x_r], target (x - x_r) / sigma_r, v-prediction, sampled like B
# varhead       (VH, Phase 2a): input [x_r, x_r] at t = REGRESS_T, output log sigma^2 of (x - x_r) / sigma_r per
#                     element, Gaussian NLL, one deterministic call: the variance-head baseline around the same centre
XR_MODES = ('residual_reg', 'varhead')                                  # the x0_hat slot holds x_r, ds.sigma_d is sigma_r
CENTRED_MODES = ('residual', 'regress', 'residual_reg', 'varhead')      # target (x - centre) / sigma

REGRESS_T = 999     # the regressor's constant time input: B's one-step (ddim1) call is at t = 999, where x_t is pure noise
LOGVAR_RANGE = (-10.0, 10.0)      # varhead's log-variance is clamped here before the NLL (sigma between e^-5 and e^5)


def build_residual_model(mode, base_channels, spectral_channels=64, sensor_channels=30):
    """The one place that maps a mode name to an architecture; training and evaluation both call it."""
    from model.u2net_hyperspectral import U2NetHyperspectral      # local import: u2net_hyperspectral must not import this module
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
    use_x0_hat = mode in ('residual', 'standard_xhat', 'regress') + XR_MODES
    net = U2NetHyperspectral(spectral_channels=spectral_channels, sensor_channels=sensor_channels,
                             base_channels=base_channels, extra_in_channels=spectral_channels if use_x0_hat else 0)
    return ResidualWrapper(net, use_x0_hat=use_x0_hat)


def target_for(mode, x, x0_hat, sigma_d):
    """The training target per mode (all on the loader's [-1, 1] scale); for XR_MODES x0_hat is x_r and sigma_d is sigma_r."""
    if mode in CENTRED_MODES:
        return (x - x0_hat) / sigma_d
    return x


def regress_forward(model, cond):
    """The regressor's single call: a zero x_t placeholder (the 64 channels B reserves for x_t) at t = REGRESS_T.
    Training, reconstruction and the y-swap all call this, so the placeholder and t cannot drift apart."""
    xh = cond['x0_hat']
    t = torch.full((xh.shape[0],), REGRESS_T, dtype=torch.long, device=xh.device)
    return model(torch.zeros_like(xh), cond, t)


def varhead_forward(model, cond):
    """The variance head's single call: the centre x_r itself in the x_t slot (spec §4.5), at t = REGRESS_T.
    Returns the log-variance of the standardised residual, [B, 64, H, W], unclamped."""
    xr = cond['x0_hat']
    t = torch.full((xr.shape[0],), REGRESS_T, dtype=torch.long, device=xr.device)
    return model(xr, cond, t)


def varhead_nll(logvar, r):
    """Per-element Gaussian negative log-likelihood 0.5 (log s^2 + r^2 / s^2) of the standardised residual r under
    N(0, s^2), log s^2 clamped to LOGVAR_RANGE (the 0.5 log 2 pi constant is dropped)."""
    lv = logvar.clamp(*LOGVAR_RANGE)
    return 0.5 * (lv + r ** 2 * torch.exp(-lv))


def compose(centre, sigma, r):
    """x_hat = clamp(centre + sigma * r, -1, 1): one composition rule for the regressor, B and the Phase 2a samples."""
    return (centre + sigma * r).clamp(-1, 1)
```

`model/varhead_trainer.py`:

```python
"""Objective of the variance-head baseline (mode 'varhead', spec §4.5), duck-typed to DiffusionTrainer.get_loss like
model.regression_trainer.RegressionTrainer so the training script's loops run it unchanged."""
from model.residual_wrapper import REGRESS_T, varhead_forward, varhead_nll


class VarheadTrainer:
    prediction_type = 'direct'
    loss_type = 'nll'

    def __init__(self, *, device, n_timesteps=1000):
        self.device = device
        self.n_timesteps = int(n_timesteps)       # provenance only
        self.last_nfe = 1

    def get_loss(self, model, x_0, cond=None, t=None, *, loss_in_fp32=False):
        """x_0 is the standardised residual r = (x - x_r) / sigma_r; the mean is fixed at x_r, the network predicts
        log sigma^2 per element. Returns (mean over samples of the per-sample mean NLL, info)."""
        x_0 = x_0.to(self.device)
        logvar = varhead_forward(model, cond)
        lv, r = (logvar.float(), x_0.float()) if loss_in_fp32 else (logvar, x_0)
        nll = varhead_nll(lv, r)
        loss = nll.reshape(nll.shape[0], -1).mean(dim=1).mean()
        return loss, dict(loss=loss.detach(), logvar=logvar.detach(), t=REGRESS_T)
```

- [ ] **Step 4: Run the tests and the existing mode tests**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_uq_modes.py tests/test_regress_mode.py tests/test_wrapper.py tests/test_model_extra_in.py -q`
Expected: all pass (the existing `test_regress_mode.py::test_regress_mode_shares_bs_architecture_and_target` asserts `MODES`; if it asserts the old 4-tuple literally, update that one assertion to the new 6-tuple — nothing else).

- [ ] **Step 5: Commit**

```bash
git add model/residual_wrapper.py model/varhead_trainer.py tests/test_uq_modes.py tests/test_regress_mode.py
git commit -F <message file>   # "feat(uq): modes residual_reg and varhead — B's architecture around the cached regressor centre, Gaussian-NLL variance head (Phase 2a Task 2)" + trailers
```

---
### Task 3: The centre cache dataset

**Files:**
- Create: `data_loader/xr_cache.py`
- Modify: `data_loader/hfd_residual.py` (add `raw_item`)
- Test: `tests/test_xr_cache.py`

**Interfaces:**
- Consumes: `HFD_data.__getitem__` (returns `(x64 [H, W, 64], y_d [h, w, 30])` in image mode), `residual_collate_fn`.
- Produces: `META_NAME = 'meta.json'`; `cache_path(cache_dir, split, relpath) -> str` (`<cache_dir>/<split>/<relpath without .mat>.npy`); `save_xr(cache_dir, split, relpath, x_r_hwc: np.ndarray) -> str` (float16); `write_meta(cache_dir, meta)`, `read_meta(cache_dir) -> dict`, `meta_sha256(cache_dir) -> str`; `check_crossfit_meta(meta) -> None` (raises ValueError unless `meta['half1']['train_half'] == 1`, `meta['half2']['train_half'] == 2`, `meta['full']['train_half'] == 0`); `class XRCachedData(base, cache_dir, split, data_root)` with `.d`, `.sigma_d` (= σ_R), `.sigma_r`, `.prior`, `.meta`, `.img_list` (property bound to `base.img_list`), `.path_of(file)`, `.check_files()`, `.checkpoint_extra() -> dict(sigma_r, xr_cache_sha256, centre, split_sha256)`, `__getitem__ -> (x64, y_d, x_r)` float32 HWC; `xr_collate_fn = residual_collate_fn`; `HFDResidualData.raw_item(idx) -> (x64, y_d)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_xr_cache.py`:

```python
"""Phase 2a Task 3: the centre cache stores x_r per cube as float16 next to a meta that names the regressors, sigma_r
and the split; XRCachedData substitutes x_r for x0_hat and sigma_r for sigma_d and refuses a cache of another stride,
a cache whose halves are crossed, and missing files."""
import hashlib
import json
import os

import numpy as np
import pytest
import torch

from data_loader.hfd_residual import residual_collate_fn
from data_loader.xr_cache import (META_NAME, XRCachedData, cache_path, check_crossfit_meta, meta_sha256, read_meta,
                                  save_xr, write_meta, xr_collate_fn)

ROOT = '/fake/data root'


class StubBase:
    """What XRCachedData needs from HFDResidualData: d, prior, img_list and raw_item(idx) -> (x64, y_d) HWC."""
    def __init__(self, n=4, H=8, d=4):
        g = torch.Generator().manual_seed(0)
        self.d, self.prior = d, {'s': 3e-7}
        self.img_list = [os.path.join(ROOT, f'MatFlower60/Train/P00{1 + i % 2}/{i}_1.mat') for i in range(n)]
        self.x = (torch.rand(n, H, H, 64, generator=g) * 2 - 1).numpy()
        self.y = torch.rand(n, H // d, H // d, 30, generator=g).numpy()

    def raw_item(self, idx):
        return self.x[idx], self.y[idx]


def meta_for(d=4, **over):
    m = dict(d=d, split_sha256='abc', sigma_r=0.09, sigma_full=0.095, sigma_ratio=1.0556, dtype='float16',
             half1=dict(run='regress_l1_half1_d4_b64', epoch=12, sha256='h1', train_half=1),
             half2=dict(run='regress_l1_half2_d4_b64', epoch=14, sha256='h2', train_half=2),
             full=dict(run='regress_l1_sel_d4_b64', epoch=17, sha256='f', train_half=0))
    m.update(over)
    return m


def build_cache(tmp_path, base, meta, split='train', skip=()):
    cd = str(tmp_path / 'xr_cache')
    write_meta(cd, meta)
    xr = {}
    for i, f in enumerate(base.img_list):
        rel = os.path.relpath(f, ROOT)
        xr[f] = np.clip(base.x[i] + 0.09 * np.random.default_rng(i).standard_normal(base.x[i].shape), -1, 1).astype(np.float32)
        if rel not in skip:
            save_xr(cd, split, rel, xr[f])
    return cd, xr


def test_cache_path_layout():
    assert cache_path('/c', 'train', 'MatFlower60/Train/P001/3_2.mat') == '/c/train/MatFlower60/Train/P001/3_2.npy'


def test_meta_round_trip_and_sha(tmp_path):
    cd = str(tmp_path / 'c')
    write_meta(cd, meta_for())
    assert read_meta(cd) == meta_for()
    raw = open(os.path.join(cd, META_NAME), 'rb').read()
    assert meta_sha256(cd) == hashlib.sha256(raw).hexdigest()
    with pytest.raises(FileExistsError):
        write_meta(cd, meta_for())


def test_crossfit_meta_check():
    check_crossfit_meta(meta_for())
    with pytest.raises(ValueError, match='half1'):
        check_crossfit_meta(meta_for(half1=dict(run='r', epoch=1, sha256='x', train_half=2)))
    with pytest.raises(ValueError, match='full'):
        check_crossfit_meta(meta_for(full=dict(run='r', epoch=1, sha256='x', train_half=1)))


def test_items_substitute_the_cached_centre_and_sigma_r(tmp_path):
    base = StubBase()
    cd, xr = build_cache(tmp_path, base, meta_for())
    ds = XRCachedData(base, cd, 'train', ROOT)
    ds.check_files()
    assert len(ds) == 4 and ds.d == 4 and ds.sigma_d == 0.09 and ds.sigma_r == 0.09 and ds.prior is base.prior
    x, y, r = ds[1]
    assert x.dtype == y.dtype == r.dtype == np.float32
    assert np.array_equal(x, base.x[1]) and np.array_equal(y, base.y[1])
    assert np.abs(r - xr[base.img_list[1]]).max() <= 1e-3           # float16 storage of a [-1, 1] value
    bx, by, br = xr_collate_fn([ds[0], ds[1]])
    assert xr_collate_fn is residual_collate_fn and bx.shape == (2, 64, 8, 8) and by.shape == (2, 30, 2, 2) and br.shape == (2, 64, 8, 8)


def test_img_list_is_shared_with_the_base_so_filters_apply_to_both(tmp_path):
    base = StubBase()
    cd, _ = build_cache(tmp_path, base, meta_for())
    ds = XRCachedData(base, cd, 'train', ROOT)
    ds.img_list = ds.img_list[:2]
    assert len(ds) == 2 and base.img_list == ds.img_list
    x, _, _ = ds[1]
    assert np.array_equal(x, base.x[1])


def test_checkpoint_extra_names_sigma_r_cache_sha_centre_and_split(tmp_path):
    base = StubBase()
    cd, _ = build_cache(tmp_path, base, meta_for())
    ds = XRCachedData(base, cd, 'train', ROOT)
    e = ds.checkpoint_extra()
    assert e == dict(sigma_r=0.09, xr_cache_sha256=meta_sha256(cd), centre=meta_for()['full'], split_sha256='abc')


def test_a_cache_of_another_stride_is_refused(tmp_path):
    base = StubBase(d=8)
    cd, _ = build_cache(tmp_path, base, meta_for(d=4))
    with pytest.raises(ValueError, match='d=8'):
        XRCachedData(base, cd, 'train', ROOT)


def test_a_crossed_cache_is_refused(tmp_path):
    base = StubBase()
    cd, _ = build_cache(tmp_path, base, meta_for(half2=dict(run='r', epoch=1, sha256='x', train_half=1)))
    with pytest.raises(ValueError, match='half2'):
        XRCachedData(base, cd, 'train', ROOT)


def test_missing_files_are_reported_by_check_files(tmp_path):
    base = StubBase()
    cd, _ = build_cache(tmp_path, base, meta_for(), skip=('MatFlower60/Train/P002/1_1.mat',))
    ds = XRCachedData(base, cd, 'train', ROOT)
    with pytest.raises(FileNotFoundError, match='1_1'):
        ds.check_files()
    ds.img_list = [f for f in ds.img_list if '1_1' not in f]
    ds.check_files()


def test_a_missing_meta_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match=META_NAME):
        read_meta(str(tmp_path / 'nothing'))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_xr_cache.py -q`
Expected: `ModuleNotFoundError: No module named 'data_loader.xr_cache'`.

- [ ] **Step 3: Implement**

Append to `data_loader/hfd_residual.py` inside `class HFDResidualData` (after `__getitem__`):

```python
    def raw_item(self, idx):
        """(x64, y_d) without the x0_hat estimate: the Phase 2a cache dataset reads the centre from disk instead."""
        x64, y_d = HFD_data.__getitem__(self, idx)
        return np.asarray(x64, dtype=np.float32), np.asarray(y_d, dtype=np.float32)
```

`data_loader/xr_cache.py`:

```python
"""The regressor-centre cache of Phase 2a (spec §4.3) and the dataset that reads it.

Layout: <cache_dir>/meta.json and <cache_dir>/<split>/<relpath without .mat>.npy, one float16 [H, W, 64] array per
cube: for split 'train' the estimate x_r of the half-data regressor that did NOT see the cube (cross-fitting), for
split 'val' the full selected regressor's estimate. meta.json names the three regressors (run, epoch, sha256,
train_half), sigma_r (RMS of x - x_r over the cross-fitted training cubes), sigma_full (the same on the monitoring
files with the full regressor) and the split's sha256.

XRCachedData wraps an HFDResidualData and returns (x64, y_d, x_r) with .sigma_d = sigma_r, so the trainer's loops see
B's interface with the regressor's estimate in the x0_hat slot (plan decision D1).
"""
import hashlib
import json
import os

import numpy as np
import torch

from data_loader.hfd_residual import residual_collate_fn

META_NAME = 'meta.json'
xr_collate_fn = residual_collate_fn


def cache_path(cache_dir, split, relpath):
    return os.path.join(cache_dir, split, os.path.splitext(relpath)[0] + '.npy')


def save_xr(cache_dir, split, relpath, x_r_hwc):
    path = cache_path(cache_dir, split, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp.npy'
    np.save(tmp, np.asarray(x_r_hwc, dtype=np.float16))
    os.replace(tmp, path)
    return path


def write_meta(cache_dir, meta):
    path = os.path.join(cache_dir, META_NAME)
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; a cache is built once (delete the directory to rebuild)')
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def read_meta(cache_dir):
    path = os.path.join(cache_dir, META_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found: not a centre cache (scripts/uq/build_xr_cache.py writes {META_NAME})')
    with open(path) as f:
        return json.load(f)


def meta_sha256(cache_dir):
    with open(os.path.join(cache_dir, META_NAME), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_crossfit_meta(meta):
    """The train half of each regressor must be what its role requires, or a cube was centred by a net that saw it."""
    want = dict(half1=1, half2=2, full=0)
    for key, th in want.items():
        got = (meta.get(key) or {}).get('train_half')
        if got != th:
            raise ValueError(f"cache meta {key}: train_half {got!r}, expected {th} (a '{key}' regressor must be trained "
                             f"on {'half ' + str(th) if th else 'the full training set'})")


class XRCachedData(torch.utils.data.Dataset):
    def __init__(self, base, cache_dir, split, data_root):
        self.base, self.cache_dir, self.split, self.data_root = base, cache_dir, split, data_root
        self.meta = read_meta(cache_dir)
        if int(self.meta['d']) != int(base.d):
            raise ValueError(f'cache {cache_dir} was built at d={int(self.meta["d"])} but the dataset is at d={int(base.d)}')
        check_crossfit_meta(self.meta)
        self.d = int(base.d)
        self.sigma_r = float(self.meta['sigma_r'])
        self.sigma_d = self.sigma_r            # the residual scale of THIS dataset's centre; the trainer reads ds.sigma_d
        self.prior = base.prior

    @property
    def img_list(self):
        return self.base.img_list

    @img_list.setter
    def img_list(self, value):
        self.base.img_list = value

    def path_of(self, file):
        return cache_path(self.cache_dir, self.split, os.path.relpath(file, self.data_root))

    def check_files(self):
        missing = [f for f in self.img_list if not os.path.isfile(self.path_of(f))]
        if missing:
            raise FileNotFoundError(f'{len(missing)} of {len(self.img_list)} cubes have no cached centre under '
                                    f'{self.cache_dir}/{self.split}, e.g. {os.path.relpath(missing[0], self.data_root)}')

    def checkpoint_extra(self):
        return dict(sigma_r=self.sigma_r, xr_cache_sha256=meta_sha256(self.cache_dir), centre=self.meta['full'],
                    split_sha256=self.meta['split_sha256'])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        x64, y_d = self.base.raw_item(idx)
        x_r = np.load(self.path_of(self.img_list[idx])).astype(np.float32)
        return x64, y_d, x_r
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_xr_cache.py tests/test_residual_dataset.py -q`
Expected: all pass (the dataset-marked ones skip without the dataset).

- [ ] **Step 5: Commit**

```bash
git add data_loader/xr_cache.py data_loader/hfd_residual.py tests/test_xr_cache.py
git commit -F <message file>   # "feat(uq): regressor-centre cache dataset (float16 x_r per cube, meta with sigma_r and the three regressors) (Phase 2a Task 3)" + trailers
```

---

### Task 4: Trainer integration (`--train_half`, `--xr_cache`, the two modes, meta)

**Files:**
- Modify: `main_2d_residual_fsdp.py` (imports; `get_parser`; `make_datasets`; `check_mode_args`; `make_trainer`; `checkpoint_meta`; `check_resume_meta`; `build_everything`; `run_epochs` meta call)
- Test: `tests/test_uq_training.py`

**Interfaces:**
- Consumes: Task 1 `read_split`, `SPLIT_JSON`; Task 2 `XR_MODES`, `VarheadTrainer`; Task 3 `XRCachedData`.
- Produces: CLI flags `--train_half {0,1,2}` (default 0), `--xr_cache DIR` (default ''), `--split_json PATH` (default `SPLIT_JSON`), `--loss_type {l1,l2,nll}`; `check_mode_args(args)` rules below; `checkpoint_meta(args, sigma_d, prior_s, prior_sha256, extra=None)` adds `train_half` and the `extra` dict; `check_resume_meta(ck, args, sigma_d, prior_sha256, path, warn, cache_sha=None)` also compares `meta['args']['train_half']` and, when `cache_sha` is given, `meta['xr_cache_sha256']`.

Rules of `check_mode_args`: modes `regress` and `varhead` need `--prediction_type direct` and only they may have it; `varhead` needs `--loss_type nll` and only it may have it; `--xr_cache` is required by and only allowed with `XR_MODES`; `--train_half` ≠ 0 is allowed with `--mode regress` only.

- [ ] **Step 1: Write the failing tests**

`tests/test_uq_training.py`:

```python
"""Phase 2a Task 4: the trainer accepts the two new modes and the half/cache flags, refuses the contradictory
combinations, records sigma_r / cache sha / centre / train_half in the checkpoint meta, refuses to resume against a
rebuilt cache, and trains both modes on a stub cache dataset (CPU)."""
import json
import os
from types import SimpleNamespace

import pytest
import torch

import main_2d_residual_fsdp as tr
from model.diffusion_trainer import DiffusionTrainer
from model.residual_wrapper import build_residual_model
from model.varhead_trainer import VarheadTrainer

CPU = torch.device('cpu')


class StubXRSet(torch.utils.data.Dataset):
    """(x64, y_d, x_r) HWC float32 with a known residual r = (x - x_r) / sigma_r = 0.5 * x_r, plus what the trainer
    reads from an XRCachedData: sigma_d (= sigma_r), prior, img_list, checkpoint_extra()."""
    sigma_r = sigma_d = 0.05
    prior = {'s': 3e-7}

    def __init__(self, n=8, H=16, d=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.xr = torch.rand(n, H, H, 64, generator=g) * 1.6 - 0.8
        self.x = (self.xr + self.sigma_r * 0.5 * self.xr).clamp(-1, 1)
        self.y = torch.rand(n, H // d, H // d, 30, generator=g)
        self.img_list = [f'MatFlower60/Train/P0{22 + i % 3}/f{i}_1.mat' for i in range(n)]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        return self.x[i].numpy(), self.y[i].numpy(), self.xr[i].numpy()

    def checkpoint_extra(self):
        return dict(sigma_r=self.sigma_r, xr_cache_sha256='stubsha', centre=dict(run='stub', epoch=1, sha256='c', train_half=0), split_sha256='s')


def args_for(tmp_path, mode, *extra):
    base = ['--mode', mode, '--device', 'cpu', '--base_channels', '8', '--batch_size', '2', '--num_workers', '0',
            '--num_epochs', '2', '--save_dir', str(tmp_path / 'run'), '--no_bf16', '--save_every', '1', '--val_every', '0']
    return tr.get_parser().parse_args(base + list(extra))


@pytest.mark.parametrize('mode, extra, ok', [
    ('residual_reg', ('--xr_cache', '/c'), True),
    ('residual_reg', (), False),                                               # cache required
    ('residual_reg', ('--xr_cache', '/c', '--prediction_type', 'direct'), False),
    ('residual_reg', ('--xr_cache', '/c', '--loss_type', 'nll'), False),       # nll is varhead's
    ('varhead', ('--xr_cache', '/c', '--prediction_type', 'direct', '--loss_type', 'nll'), True),
    ('varhead', ('--xr_cache', '/c', '--prediction_type', 'direct'), False),   # needs nll
    ('varhead', ('--xr_cache', '/c', '--prediction_type', 'v', '--loss_type', 'nll'), False),
    ('varhead', ('--prediction_type', 'direct', '--loss_type', 'nll'), False),
    ('regress', ('--prediction_type', 'direct', '--train_half', '1'), True),
    ('regress', ('--prediction_type', 'direct', '--xr_cache', '/c'), False),   # cache only for XR modes
    ('residual', ('--train_half', '1'), False),                                # halves are for regressors
    ('residual', (), True),
])
def test_check_mode_args(tmp_path, mode, extra, ok):
    a = args_for(tmp_path, mode, *extra)
    if ok:
        tr.check_mode_args(a)
    else:
        with pytest.raises(ValueError):
            tr.check_mode_args(a)


def test_make_trainer_for_the_new_modes(tmp_path):
    a = args_for(tmp_path, 'varhead', '--xr_cache', '/c', '--prediction_type', 'direct', '--loss_type', 'nll')
    assert isinstance(tr.make_trainer(a, CPU), VarheadTrainer)
    a = args_for(tmp_path, 'residual_reg', '--xr_cache', '/c')
    t = tr.make_trainer(a, CPU)
    assert isinstance(t, DiffusionTrainer) and t.prediction_type == 'v'


def test_checkpoint_meta_carries_train_half_and_the_cache_extra(tmp_path):
    a = args_for(tmp_path, 'regress', '--prediction_type', 'direct', '--train_half', '2')
    m = tr.checkpoint_meta(a, 0.05, 3e-7, 'psha')
    assert m['train_half'] == 2 and 'sigma_r' not in m
    a = args_for(tmp_path, 'residual_reg', '--xr_cache', '/c')
    m = tr.checkpoint_meta(a, 0.09, 3e-7, 'psha', extra=StubXRSet().checkpoint_extra())
    assert m['sigma_d'] == 0.09 and m['sigma_r'] == 0.05 and m['xr_cache_sha256'] == 'stubsha' and m['centre']['run'] == 'stub' and m['train_half'] == 0


def test_resume_refuses_a_rebuilt_cache_and_another_half(tmp_path):
    a = args_for(tmp_path, 'residual_reg', '--xr_cache', '/c')
    ck = dict(meta=tr.checkpoint_meta(a, 0.05, 3e-7, 'psha', extra=StubXRSet().checkpoint_extra()))
    assert tr.check_resume_meta(ck, a, 0.05, 'psha', cache_sha='stubsha', warn=False)
    with pytest.raises(ValueError, match='xr_cache_sha256'):
        tr.check_resume_meta(ck, a, 0.05, 'psha', cache_sha='othersha', warn=False)
    a1 = args_for(tmp_path, 'regress', '--prediction_type', 'direct', '--train_half', '1')
    ck1 = dict(meta=tr.checkpoint_meta(a1, 0.05, 3e-7, 'psha'))
    a2 = args_for(tmp_path, 'regress', '--prediction_type', 'direct', '--train_half', '2')
    with pytest.raises(ValueError, match='train_half'):
        tr.check_resume_meta(ck1, a2, 0.05, 'psha', warn=False)


def parts_for(args, ds):
    loader = tr.make_loader(ds, args, batch_size=args.batch_size, shuffle=True, rank=0, world=1, device=CPU)
    model = build_residual_model(args.mode, args.base_channels)
    trainer = tr.make_trainer(args, CPU)
    opt = tr.make_optimizer(model, args)
    return ds, loader, model, trainer, opt, tr.make_scheduler(opt, args)


@pytest.mark.parametrize('mode, extra', [('residual_reg', ('--xr_cache', '/c')),
                                         ('varhead', ('--xr_cache', '/c', '--prediction_type', 'direct', '--loss_type', 'nll'))])
def test_both_modes_train_on_the_stub_cache_and_write_the_meta(tmp_path, mode, extra):
    a = args_for(tmp_path, mode, *extra)
    os.makedirs(a.save_dir)
    ds = StubXRSet()
    tr.run_epochs(a, parts_for(a, ds), tr.NO_DIST, CPU)
    log = [json.loads(l) for l in open(os.path.join(a.save_dir, 'train_log.jsonl'))]
    assert [r['epoch'] for r in log] == [1, 2] and all(torch.isfinite(torch.tensor(r['loss'])) for r in log)
    ck = torch.load(os.path.join(a.save_dir, 'checkpoint_epoch_2.pth'), map_location='cpu', weights_only=False)
    m = ck['meta']
    assert m['mode'] == mode and m['sigma_d'] == 0.05 and m['sigma_r'] == 0.05 and m['xr_cache_sha256'] == 'stubsha' and m['centre']['run'] == 'stub'
    if mode == 'varhead':
        assert log[1]['loss'] < log[0]['loss']
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_uq_training.py -q`
Expected: failures on `--train_half` (unrecognized argument) and friends.

- [ ] **Step 3: Implement**

In `main_2d_residual_fsdp.py`:

(a) Imports — extend the existing `from model.residual_wrapper import ...` line to also import `XR_MODES`, and add:

```python
from model.varhead_trainer import VarheadTrainer
from data_loader.xr_cache import XRCachedData
from data_loader.warmstart_paths import SPLIT_JSON
from scripts.uq.split_halves import read_split
```

(b) `get_parser`: change `--loss_type` to `choices=['l1', 'l2', 'nll']` and add after it:

```python
    ap.add_argument('--train_half', type=int, choices=[0, 1, 2], default=0,
                    help='Phase 2a: train a regressor on half 1 or 2 of the cross-fitting split only (0 = all files)')
    ap.add_argument('--split_json', default=SPLIT_JSON, help='the cross-fitting split (scripts/uq/split_halves.py)')
    ap.add_argument('--xr_cache', default='', help='Phase 2a: regressor-centre cache dir (required by --mode residual_reg / varhead)')
```

(c) Replace `check_mode_args` and `make_trainer`:

```python
def check_mode_args(args):
    """regress and varhead are the one-call modes (--prediction_type direct); varhead's loss is the Gaussian NLL
    (--loss_type nll, decision D2); the XR modes read their centre from --xr_cache; --train_half is for half regressors."""
    if (args.mode in ('regress', 'varhead')) != (args.prediction_type == 'direct'):
        raise ValueError(f"--mode regress / varhead need --prediction_type direct, and 'direct' needs one of them "
                         f"(got mode={args.mode!r}, prediction_type={args.prediction_type!r})")
    if (args.mode == 'varhead') != (args.loss_type == 'nll'):
        raise ValueError(f"--mode varhead needs --loss_type nll, and 'nll' needs --mode varhead "
                         f"(got mode={args.mode!r}, loss_type={args.loss_type!r})")
    if (args.mode in XR_MODES) != bool(args.xr_cache):
        raise ValueError(f'--xr_cache is required by, and only allowed with, --mode {" / ".join(XR_MODES)} '
                         f'(got mode={args.mode!r}, xr_cache={args.xr_cache!r})')
    if args.train_half and args.mode != 'regress':
        raise ValueError(f'--train_half is for the half-data regressors (--mode regress) only, got mode={args.mode!r}')


def make_trainer(args, device):
    check_mode_args(args)
    if args.mode == 'regress':
        return RegressionTrainer(device=device, loss_type=args.loss_type, n_timesteps=N_TIMESTEPS)
    if args.mode == 'varhead':
        return VarheadTrainer(device=device, n_timesteps=N_TIMESTEPS)
    return DiffusionTrainer(device=device, n_timesteps=N_TIMESTEPS, loss_type=args.loss_type,
                            prediction_type=args.prediction_type, snr_gamma=args.snr_gamma if args.snr_gamma > 0 else None)
```

(d) `make_datasets`: between the gate-filter `assert` and the `if args.limit_files:` block insert the half filter and the wrapping, and add the file check at the end:

```python
    if args.train_half:
        half = set(read_split(args.split_json)[f'half{args.train_half}'])
        n_before = len(train.img_list)
        train.img_list = [f for f in train.img_list if os.path.relpath(f, args.data_root) in half]
        if len(train.img_list) != len(half):
            raise ValueError(f'--train_half {args.train_half}: {len(train.img_list)} of the split\'s {len(half)} files are in the '
                             f'{n_before}-file training list (split {args.split_json} vs data_root {args.data_root})')
    if args.xr_cache:
        train = XRCachedData(train, args.xr_cache, 'train', args.data_root)
        val = XRCachedData(val, args.xr_cache, 'val', args.data_root)
    if args.limit_files:
        train.img_list = train.img_list[:args.limit_files]; val.img_list = val.img_list[:max(8, args.limit_files // 8)]
    if args.xr_cache:
        train.check_files(); val.check_files()
    return train, val
```

(e) `checkpoint_meta` gains `extra=None`:

```python
def checkpoint_meta(args, sigma_d, prior_s, prior_sha256, extra=None):
    meta = dict(mode=args.mode, d=args.d, base_channels=args.base_channels, prediction_type=args.prediction_type,
                n_timesteps=N_TIMESTEPS, sigma_d=float(sigma_d), prior_s=float(prior_s), prior_sha256=prior_sha256, args=vars(args),
                git_head=git_head(), torch=torch.__version__, train_half=int(getattr(args, 'train_half', 0)))
    if extra:
        meta.update(extra)                 # XR modes: sigma_r, xr_cache_sha256, centre, split_sha256 (XRCachedData.checkpoint_extra)
    return meta
```

(f) `check_resume_meta(ck, args, sigma_d, prior_sha256, path='<checkpoint>', warn=True, cache_sha=None)`: change the args loop to `for k in ('loss_type', 'seed', 'train_half'):` and before `if bad:` add:

```python
    if cache_sha is not None and meta.get('xr_cache_sha256') != cache_sha:
        bad.append(f'xr_cache_sha256: checkpoint {meta.get("xr_cache_sha256", "<missing>")!r} vs cache {cache_sha!r} (the cache was rebuilt)')
```

(g) `build_everything`: replace the `check_resume_meta(...)` call with

```python
        cache_sha = train_ds.checkpoint_extra()['xr_cache_sha256'] if args.xr_cache else None
        check_resume_meta(ck, args, train_ds.sigma_d, file_sha256(args.prior), path=resume_path, warn=dist_ctx.rank == 0, cache_sha=cache_sha)
```

(h) `run_epochs`: replace the `meta = checkpoint_meta(...)` line with

```python
    ds_meta = parts[0] if isinstance(parts, tuple) else parts.train_ds
    extra = ds_meta.checkpoint_extra() if hasattr(ds_meta, 'checkpoint_extra') else None
    meta = checkpoint_meta(args, sigma_d, ds_meta.prior['s'], file_sha256(args.prior) if os.path.exists(args.prior) else '', extra=extra)
```

- [ ] **Step 4: Run the new and the existing trainer tests**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_uq_training.py tests/test_regress_mode.py tests/test_train_residual.py -q`
Expected: all pass. (`test_regress_mode.py::test_make_trainer_ties_mode_regress_to_prediction_type_direct` still holds under the new rule.)

- [ ] **Step 5: Commit**

```bash
git add main_2d_residual_fsdp.py tests/test_uq_training.py
git commit -F <message file>   # "feat(uq): trainer learns --train_half / --xr_cache, modes residual_reg and varhead, cache provenance in the checkpoint meta (Phase 2a Task 4)" + trailers
```

---
### Task 5: The cache builder

**Files:**
- Create: `scripts/uq/build_xr_cache.py`
- Test: `tests/test_build_xr_cache.py`

**Interfaces:**
- Consumes: Task 1 `read_split`, `UQ_DIR`; Task 2 `compose`, `regress_forward`, `build_residual_model`; Task 3 `save_xr`, `write_meta`; `HFDResidualData`, `residual_collate_fn`, `limit_worker_threads`, `SUBSET`, `file_sha256`, `git_head`.
- Produces: `ckpt_info(path, role, want_half, d) -> (ck, info)` with `info = dict(path, run, epoch, sha256, train_half, base_channels, sigma_d)`; `load_regressor(ck, base_channels, device) -> model`; `half_of(relpath, half1: set, half2: set) -> 1 | 2` (raises KeyError otherwise); `centre_batch(model, y, xh, sigma_d) -> x_r [B, 64, H, W]`; `run(args, make_ds=None) -> meta` where `make_ds(split_name) -> dataset` defaults to the real `HFDResidualData`; CLI `python scripts/uq/build_xr_cache.py --d 8 --half1 CKPT --half2 CKPT --full CKPT [--out UQ_DIR/d8/xr_cache] [--split_json SPLIT_JSON] [--device cuda:0] [--batch_size 16] [--num_workers 8] [--limit_files 0]`; output meta keys: `d, split_path, split_sha256, half1, half2, full, sigma_r, sigma_full, sigma_ratio, sigma_flag, n_train, n_val, n_val_monitoring, dtype, git_head, created, limit_files`.

- [ ] **Step 1: Write the failing tests**

`tests/test_build_xr_cache.py`:

```python
"""Phase 2a Task 5: the cache builder refuses a regressor whose train_half does not match its role, centres every
training cube with the OTHER half's regressor and every test-split cube with the full one, stores float16, and writes
sigma_r (RMS over the cross-fitted training residuals) and sigma_full (monitoring files only) into the meta."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_loader.xr_cache import cache_path, read_meta
from model.residual_wrapper import build_residual_model
from scripts.uq import build_xr_cache as bx
from scripts.uq.split_halves import write_split

ROOT = '/fake/root'
GATE = ['MatFlower60/Train/P023/0_1.mat']       # one "gate" file in the stub test split (excluded from sigma_full)


class StubDS:
    """(x64, y_d, x0_hat) HWC float32; the residual x - x0_hat has RMS 0.1 by construction."""
    sigma_d = 0.1
    prior = {'s': 3e-7}
    d = 4

    def __init__(self, files, seed):
        g = torch.Generator().manual_seed(seed)
        n = len(files)
        self.img_list = [os.path.join(ROOT, f) for f in files]
        self.xh = torch.rand(n, 8, 8, 64, generator=g) * 1.2 - 0.6
        self.x = self.xh + 0.1 * torch.randn(n, 8, 8, 64, generator=g).sign()      # |x - xh| = 0.1 exactly
        self.y = torch.rand(n, 2, 2, 30, generator=g)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        return self.x[i].numpy(), self.y[i].numpy(), self.xh[i].numpy()


def fake_ckpt(tmp_path, name, train_half, d=4, mode='regress', scale=0.0):
    """A regressor checkpoint whose net predicts `scale` everywhere (a constant residual), with the trainer's meta."""
    m = build_residual_model('regress', 8)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        last_bias = [p for n, p in m.net.named_parameters() if n.endswith('.bias')][-1]
        last_bias.fill_(scale)                      # zero weights everywhere: the net's output is this constant
    out = m(torch.zeros(1, 64, 8, 8), {'sensor': torch.zeros(1, 30, 2, 2), 'x0_hat': torch.zeros(1, 64, 8, 8)}, torch.zeros(1, dtype=torch.long))
    assert torch.allclose(out, torch.full_like(out, scale)), 'the last .bias of the net is not its output bias; pick the output layer explicitly'
    run = tmp_path / 'runs' / name
    run.mkdir(parents=True)
    path = run / 'checkpoint_epoch_3.pth'
    meta = dict(mode=mode, d=d, base_channels=8, prediction_type='direct', sigma_d=0.1, prior_sha256='p',
                args=dict(train_half=train_half, loss_type='l1', seed=42), train_half=train_half)
    torch.save(dict(model_state_dict=m.state_dict(), meta=meta), path)
    return str(path)


def test_ckpt_info_checks_role_mode_and_stride(tmp_path):
    p = fake_ckpt(tmp_path, 'regress_l1_half1_d4_b64', 1)
    ck, info = bx.ckpt_info(p, 'half1', 1, 4)
    assert info['run'] == 'regress_l1_half1_d4_b64' and info['epoch'] == 3 and info['train_half'] == 1 and info['base_channels'] == 8 and len(info['sha256']) == 64
    with pytest.raises(ValueError, match='train_half'):
        bx.ckpt_info(p, 'half2', 2, 4)
    with pytest.raises(ValueError, match='d='):
        bx.ckpt_info(p, 'half1', 1, 8)
    q = fake_ckpt(tmp_path, 'residual_d4_b64', 0, mode='residual')
    with pytest.raises(ValueError, match='mode'):
        bx.ckpt_info(q, 'full', 0, 4)


def test_half_of():
    assert bx.half_of('a', {'a'}, {'b'}) == 1 and bx.half_of('b', {'a'}, {'b'}) == 2
    with pytest.raises(KeyError):
        bx.half_of('c', {'a'}, {'b'})


def test_run_crossfits_train_uses_full_on_val_and_writes_sigmas(tmp_path, monkeypatch):
    train_files = [f'MatFlower60/Train/P00{1 + i % 2}/{i}_1.mat' for i in range(6)]
    val_files = GATE + ['MatFlower60/Train/P023/1_1.mat', 'MatFlower60/Train/P024/2_1.mat']
    split = dict(seed=0, half1=sorted(train_files[0::2]), half2=sorted(train_files[1::2]), images_per_class={})
    sp = tmp_path / 'split.json'
    write_split(str(sp), split)
    # half1's net sees H1 -> must centre H2; give the three nets different constant residuals to tell them apart
    h1 = fake_ckpt(tmp_path, 'regress_l1_half1_d4_b64', 1, scale=1.0)
    h2 = fake_ckpt(tmp_path, 'regress_l1_half2_d4_b64', 2, scale=2.0)
    fu = fake_ckpt(tmp_path, 'regress_l1_sel_d4_b64', 0, scale=3.0)
    monkeypatch.setattr(bx, 'gate_relpaths', lambda data_root: set(GATE))
    ds = {'train': StubDS(train_files, 0), 'val': StubDS(val_files, 1)}
    out = str(tmp_path / 'xr_cache')
    args = SimpleNamespace(d=4, half1=h1, half2=h2, full=fu, out=out, split_json=str(sp), device='cpu', batch_size=4,
                           num_workers=0, limit_files=0, data_root=ROOT, prior='unused')
    meta = bx.run(args, make_ds=lambda split_name: ds[split_name])
    assert read_meta(out) == meta and meta['n_train'] == 6 and meta['n_val'] == 3 and meta['n_val_monitoring'] == 2
    assert meta['half1']['train_half'] == 1 and meta['half2']['train_half'] == 2 and meta['full']['train_half'] == 0
    for i, f in enumerate(train_files):
        xr = np.load(cache_path(out, 'train', f))
        want_scale = 2.0 if f in split['half1'] else 1.0              # H1 cubes are centred by the half2 net
        expect = np.clip(ds['train'].xh[i].numpy() + 0.1 * want_scale, -1, 1)
        assert xr.dtype == np.float16 and np.abs(xr.astype(np.float32) - expect).max() < 2e-3
    for i, f in enumerate(val_files):
        xr = np.load(cache_path(out, 'val', f))
        expect = np.clip(ds['val'].xh[i].numpy() + 0.1 * 3.0, -1, 1)
        assert np.abs(xr.astype(np.float32) - expect).max() < 2e-3
    # sigma_r: RMS of x - x_r over the train cubes (float32 x_r), computed independently here
    ss, n = 0.0, 0
    for i, f in enumerate(train_files):
        want_scale = 2.0 if f in split['half1'] else 1.0
        xr = np.clip(ds['train'].xh[i].numpy() + 0.1 * want_scale, -1, 1)
        ss += float(((ds['train'].x[i].numpy() - xr) ** 2).sum()); n += xr.size
    assert abs(meta['sigma_r'] - np.sqrt(ss / n)) < 1e-6
    assert meta['sigma_flag'] == (not 0.9 <= meta['sigma_ratio'] <= 1.1) and meta['dtype'] == 'float16'
    with pytest.raises(FileExistsError):
        bx.run(args, make_ds=lambda split_name: ds[split_name])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_build_xr_cache.py -q`
Expected: `ImportError: cannot import name 'build_xr_cache'`.

- [ ] **Step 3: Implement**

`scripts/uq/build_xr_cache.py`:

```python
"""Build the regressor-centre cache of one stride (spec §4.3; layout in data_loader/xr_cache.py).

    python scripts/uq/build_xr_cache.py --d 8 --half1 runs/regress_l1_half1_d8_b64/checkpoint_epoch_E.pth \
        --half2 runs/regress_l1_half2_d8_b64/checkpoint_epoch_E.pth --full runs/regress_l1_sel_d8_b64/checkpoint_epoch_11.pth

Training cubes of half 1 are centred by the half-2 regressor and vice versa (cross-fitting); every test-split cube
(monitoring + gate) is centred by the full regressor (decision D3). sigma_r = RMS of x - x_r over all cross-fitted
training cubes (float32, before the float16 cast; the definition of scripts/compute_residual_scale.py); sigma_full =
the same over the monitoring files only. A sigma_full / sigma_r ratio outside [0.9, 1.1] sets sigma_flag (reported,
not fatal). Refuses an existing cache.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from data_loader.hfd_residual import HFDResidualData, limit_worker_threads, residual_collate_fn  # noqa: E402
from data_loader.warmstart_paths import DATA_ROOT, PRIOR, SPLIT_JSON, SUBSET, UQ_DIR, file_sha256, git_head  # noqa: E402
from data_loader.xr_cache import META_NAME, save_xr, write_meta  # noqa: E402
from model.residual_wrapper import build_residual_model, compose, regress_forward  # noqa: E402
from scripts.uq.split_halves import read_split  # noqa: E402


def ckpt_info(path, role, want_half, d):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    meta = (ck.get('meta') if isinstance(ck, dict) else None) or {}
    bad = []
    if meta.get('mode') != 'regress':
        bad.append(f"mode {meta.get('mode')!r} (need 'regress')")
    if int(meta.get('d', -1)) != int(d):
        bad.append(f"d={meta.get('d')!r} (need d={d})")
    th = int((meta.get('args') or {}).get('train_half', meta.get('train_half', 0)) or 0)
    if th != want_half:
        bad.append(f'train_half {th} (a {role} regressor must have train_half {want_half})')
    if bad:
        raise ValueError(f'{role} checkpoint {path}: ' + '; '.join(bad))
    m = re.search(r'checkpoint_epoch_(\d+)\.pth$', path)
    info = dict(path=os.path.abspath(path), run=os.path.basename(os.path.dirname(os.path.abspath(path))),
                epoch=int(m.group(1)) if m else None, sha256=file_sha256(path), train_half=int(want_half),
                base_channels=int(meta['base_channels']), sigma_d=float(meta['sigma_d']))
    return ck, info


def load_regressor(ck, base_channels, device):
    model = build_residual_model('regress', base_channels)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    return model.to(device).eval()


def half_of(relpath, half1, half2):
    if relpath in half1:
        return 1
    if relpath in half2:
        return 2
    raise KeyError(f'{relpath} is in neither half of the split')


@torch.no_grad()
def centre_batch(model, y, xh, sigma_d):
    return compose(xh, sigma_d, regress_forward(model, {'sensor': y, 'x0_hat': xh}))


def gate_relpaths(data_root):
    with open(SUBSET) as f:
        return set(json.load(f)['files'])


def _loader(ds, args):
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                       collate_fn=residual_collate_fn, worker_init_fn=limit_worker_threads)


def run(args, make_ds=None):
    if os.path.exists(os.path.join(args.out, META_NAME)):
        raise FileExistsError(f'{args.out} already holds a cache; delete the directory to rebuild')
    device = torch.device(args.device)
    split = read_split(args.split_json)
    half1, half2 = set(split['half1']), set(split['half2'])
    (ck1, i1), (ck2, i2), (ckf, i_f) = (ckpt_info(args.half1, 'half1', 1, args.d), ckpt_info(args.half2, 'half2', 2, args.d),
                                        ckpt_info(args.full, 'full', 0, args.d))
    if len({i1['base_channels'], i2['base_channels'], i_f['base_channels']}) != 1:
        raise ValueError('the three regressors differ in base_channels')
    nets = {1: load_regressor(ck1, i1['base_channels'], device), 2: load_regressor(ck2, i2['base_channels'], device),
            0: load_regressor(ckf, i_f['base_channels'], device)}
    gate = gate_relpaths(args.data_root)
    if make_ds is None:
        def make_ds(split_name):
            """Production (limit_files 0): all 9 000 training cubes and all 1 000 test-split cubes. With --limit_files N
            (smoke tests): train[:N], and for val the first max(8, N // 8) NON-gate files -- the trainer's own order
            (make_datasets drops the gate cubes before applying its limit), so the two agree on which files exist."""
            ds = HFDResidualData(data_path=args.data_root, split='train' if split_name == 'train' else 'test',
                                 sensor_down_sample_rate=args.d, prior_path=args.prior)
            if args.limit_files:
                if split_name == 'train':
                    ds.img_list = ds.img_list[:args.limit_files]
                else:
                    gate_abs = {os.path.normpath(os.path.join(args.data_root, f)) for f in gate}
                    ds.img_list = [f for f in ds.img_list if os.path.normpath(f) not in gate_abs][:max(8, args.limit_files // 8)]
            return ds
    stats, counts, t0 = {}, {}, time.time()
    for split_name in ('train', 'val'):
        ds = make_ds(split_name)
        sigma_d = float(ds.sigma_d)
        rel = [os.path.relpath(f, args.data_root) for f in ds.img_list]
        ss, n, n_sigma = 0.0, 0, 0
        pos = 0
        for x, y, xh in _loader(ds, args):
            B = x.shape[0]
            files = rel[pos:pos + B]
            pos += B
            x, y, xh = x.to(device), y.to(device), xh.to(device)
            if split_name == 'train':
                which = torch.tensor([3 - half_of(f, half1, half2) for f in files], device=device)   # H1 -> net 2, H2 -> net 1
                x_r = torch.empty_like(xh)
                for k in (1, 2):
                    sel = which == k
                    if sel.any():
                        x_r[sel] = centre_batch(nets[k], y[sel], xh[sel], sigma_d)
                use = torch.ones(B, dtype=torch.bool, device=device)
            else:
                x_r = centre_batch(nets[0], y, xh, sigma_d)
                use = torch.tensor([f not in gate for f in files], device=device)      # sigma_full: monitoring files only
            if use.any():
                ss += float(((x[use] - x_r[use]).double() ** 2).sum()); n += int(x[use].numel())
                n_sigma += int(use.sum())
            for b, f in enumerate(files):
                save_xr(args.out, split_name, f, x_r[b].permute(1, 2, 0).cpu().numpy())
        stats[split_name] = float(np.sqrt(ss / max(n, 1)))
        counts[split_name] = (len(rel), n_sigma)
        print(f'{split_name}: {len(rel)} cubes cached ({time.time() - t0:.0f} s), RMS residual {stats[split_name]:.6f} over {n_sigma} cubes', flush=True)
    ratio = stats['val'] / stats['train']
    meta = dict(d=int(args.d), split_path=os.path.abspath(args.split_json), split_sha256=file_sha256(args.split_json),
                half1=i1, half2=i2, full=i_f, sigma_r=stats['train'], sigma_full=stats['val'], sigma_ratio=ratio,
                sigma_flag=bool(not 0.9 <= ratio <= 1.1), n_train=counts['train'][0], n_val=counts['val'][0],
                n_val_monitoring=counts['val'][1], dtype='float16', git_head=git_head(), created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                limit_files=int(args.limit_files))
    write_meta(args.out, meta)
    print(f"wrote {os.path.join(args.out, META_NAME)}: sigma_r {meta['sigma_r']:.6f}, sigma_full {meta['sigma_full']:.6f}, "
          f"ratio {ratio:.4f}{' FLAG (outside [0.9, 1.1])' if meta['sigma_flag'] else ''}")
    return meta


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--half1', required=True); ap.add_argument('--half2', required=True); ap.add_argument('--full', required=True)
    ap.add_argument('--out', default=None, help='default: UQ_DIR/d<d>/xr_cache')
    ap.add_argument('--split_json', default=SPLIT_JSON)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size', type=int, default=16); ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--limit_files', type=int, default=0, help='smoke tests: the first N training files and max(8, N // 8) test-split files')
    ap.add_argument('--data_root', default=DATA_ROOT); ap.add_argument('--prior', default=PRIOR)
    return ap


if __name__ == '__main__':
    a = get_parser().parse_args()
    a.out = a.out or os.path.join(UQ_DIR, f'd{a.d}', 'xr_cache')
    run(a)
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_build_xr_cache.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/uq/build_xr_cache.py tests/test_build_xr_cache.py
git commit -F <message file>   # "feat(uq): cache builder — cross-fitted centres for training cubes, full-regressor centres for the test split, sigma_r / sigma_full in the meta (Phase 2a Task 5)" + trailers
```

---

### Task 6: Launcher guards for half regressors and the new modes

**Files:**
- Modify: `scripts/pilot/launch_run.sh`
- Test: `tests/test_launch_run_name.py` (append)

**Interfaces:**
- Produces: `launch_run.sh` refuses (exit 1, nothing created) a regress launch whose `RUN_NAME` contains `_half<N>_` (or ends in `_half<N>`) when `--train_half` is not N, and a `--train_half N` (N ≠ 0) under a name without `_half<N>`; refuses a `varhead` launch without `--loss_type nll`. Accepts `residual_reg_<...>` and `varhead_<...>` names (the existing `^${MODE}_[A-Za-z0-9_]+$` rule).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_launch_run_name.py`:

```python
# ----------------------------------------------------------------------------- Phase 2a: half regressors and the new modes
D8 = ('--ds', '8')
XR = ('--xr_cache', '/tmp/xr_cache')


@pytest.mark.parametrize('run_name, mode, gpu, extra', [
    ('regress_l1_half1_d8_b64', 'regress', 0, L1 + D8 + ('--train_half', '1')),
    ('regress_l1_half2_d8_b64', 'regress', 1, L1 + D8 + ('--train_half=2',)),
    ('regress_l1_half1', 'regress', 0, L1 + ('--train_half', '1')),                   # _half<N> at the end
    ('regress_l1_sel_d8_b64', 'regress', 0, L1 + D8 + ('--train_half', '0')),         # 0 = all files, no token needed
    ('residual_reg_d8_b64', 'residual_reg', 0, D8 + XR),
    ('varhead_d8_b64', 'varhead', 1, ('--prediction_type', 'direct', '--loss_type', 'nll') + D8 + XR),
])
def test_the_phase_2a_launches_are_accepted(env, tmp_path, run_name, mode, gpu, extra):
    r = launch(env, run_name, mode, gpu, *extra)
    assert r.returncode == 0, r.stdout + r.stderr
    a = stub_lines(tmp_path, 1)[0]
    assert f'--mode {mode}' in a and a.endswith(' '.join(extra))
    wait_dead(tmp_path / 'root' / 'runs' / run_name)


@pytest.mark.parametrize('run_name, mode, extra, why', [
    ('regress_l1_half1_d8_b64', 'regress', L1 + ('--train_half', '2'), 'names half 1'),
    ('regress_l1_half1_d8_b64', 'regress', L1, 'names half 1'),
    ('regress_l1_sel_d8_b64', 'regress', L1 + ('--train_half', '2'), 'names no _half'),
    ('varhead_d8_b64', 'varhead', ('--prediction_type', 'direct') + XR, 'needs --loss_type nll'),
    ('varhead_d8_b64', 'varhead', ('--prediction_type', 'direct', '--loss_type', 'l1') + XR, 'needs --loss_type nll'),
])
def test_phase_2a_contradictions_are_refused_before_anything_starts(env, tmp_path, run_name, mode, extra, why):
    r = launch(env, run_name, mode, 0, *extra)
    assert r.returncode == 1 and why in r.stdout, r.stdout + r.stderr
    assert not (tmp_path / 'root').exists() and not (tmp_path / 'stub_args').exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_launch_run_name.py -q -k phase_2a`
Expected: the refusal cases fail (the launcher accepts them today).

- [ ] **Step 3: Implement** — in `scripts/pilot/launch_run.sh`, inside the `if [[ "$MODE" == regress ]]; then … fi` block, extend the argument scan and add the half guard after the seed guard:

```bash
  LOSS=""; SEED=""; HALF=""; PREV=""
  for A in "$@"; do                                       # the last --loss_type / --seed / --train_half wins, as in argparse
    [[ "$PREV" == --loss_type ]] && LOSS=$A
    [[ "$A" == --loss_type=* ]] && LOSS=${A#--loss_type=}
    [[ "$PREV" == --seed ]] && SEED=$A
    [[ "$A" == --seed=* ]] && SEED=${A#--seed=}
    [[ "$PREV" == --train_half ]] && HALF=$A
    [[ "$A" == --train_half=* ]] && HALF=${A#--train_half=}
    PREV=$A
  done
```

and, still inside the regress block, after the seed guard's closing `fi`:

```bash
  HALF_RE='_half([12])(_|$)'
  if [[ "${RUN_NAME:-}" =~ $HALF_RE ]]; then              # a half-data regressor (Phase 2a) must pass --train_half N
    NAME_HALF=${BASH_REMATCH[1]}
    if [[ "$HALF" != "$NAME_HALF" ]]; then
      echo "RUN_NAME='$RUN_NAME' names half $NAME_HALF but --train_half is '$HALF'; not launching"; exit 1
    fi
  elif [[ -n "$HALF" && "$HALF" != 0 ]]; then
    echo "--train_half $HALF but RUN_NAME='${RUN_NAME:-}' names no _half<N>_ (half regressors carry it); not launching"; exit 1
  fi
```

Then add a varhead block right after the regress block:

```bash
if [[ "$MODE" == varhead ]]; then
  LOSS=""; PREV=""
  for A in "$@"; do
    [[ "$PREV" == --loss_type ]] && LOSS=$A
    [[ "$A" == --loss_type=* ]] && LOSS=${A#--loss_type=}
    PREV=$A
  done
  if [[ "$LOSS" != nll ]]; then
    echo "varhead needs --loss_type nll in the extra args (got '${LOSS}'); not launching"; exit 1
  fi
fi
```

Update the header comment's usage line to `residual|standard|standard_xhat|regress|residual_reg|varhead` and add one sentence: "a regress RUN_NAME with `_half<N>` must pass `--train_half N` (and `--train_half N` needs the token); varhead needs `--loss_type nll`".

- [ ] **Step 4: Run the launcher tests**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_launch_run_name.py -q`
Expected: all pass (old and new).

- [ ] **Step 5: Commit**

```bash
git add scripts/pilot/launch_run.sh tests/test_launch_run_name.py
git commit -F <message file>   # "feat(uq): launch_run.sh guards for half regressors (_half<N> <-> --train_half) and varhead (--loss_type nll) (Phase 2a Task 6)" + trailers
```

---
### Task 7: The metrics module

**Files:**
- Create: `scripts/uq/metrics.py`
- Test: `tests/test_uq_metrics.py`

**Interfaces:**
- Consumes: numpy, torch (for `erf`).
- Produces (all numpy, float64 inside, inputs on the [0, 1] display scale unless stated):
  - `crps_samples(samples [K, ...], x [...]) -> [...]` per-element fair CRPS; raises ValueError for K < 2.
  - `crps_gaussian(mu, sigma, x) -> [...]` closed form; sigma floored at 1e-12.
  - `coverage_samples(samples, x, levels=(0.5, 0.9)) -> {level: fraction}` empirical central intervals (linear quantiles); boundary equality counts as covered.
  - `coverage_gaussian(mu, sigma, x, levels=(0.5, 0.9)) -> {level: fraction}`, `Z = {0.5: 0.6744897501960817, 0.9: 1.6448536269514722}`.
  - `sparsification_curve(score, err2, fractions=FRACTIONS) -> [50]` and `ause(pred_std, err) -> float`, `FRACTIONS = np.arange(0, 1.0, 0.02)`.
  - `spread_ratio(pred_var, err) -> float` = mean(pred_var) / mean(err²).
  - `laplacian(u) -> same shape` (3 × 3, reflect padding, last two axes), `hf_energy_ratio(x_hat, x) -> float` = log(E(x̂)/E(x)).
  - `sobel_mag(u)`, `grad_w1(x_hat, x, bins=100, vmax=0.5) -> float`.
  - `to01(t) -> np.ndarray` converting a [−1, 1] torch tensor or array to the [0, 1] scale as float64.

- [ ] **Step 1: Write the failing tests**

`tests/test_uq_metrics.py`:

```python
"""Phase 2a Task 7: every metric agrees with a closed form or a known ordering on synthetic data."""
import numpy as np
import pytest
import torch

from scripts.uq import metrics as M


def test_crps_samples_matches_the_gaussian_closed_form_for_many_samples():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    s = rng.standard_normal((2000, 5000))
    est, exact = M.crps_samples(s, x), M.crps_gaussian(np.zeros(5000), np.ones(5000), x)
    assert est.shape == (5000,) and abs(est.mean() - exact.mean()) / exact.mean() < 0.01
    assert abs(exact.mean() - (np.sqrt(2) - 1) / np.sqrt(np.pi)) < 0.01      # CRPS(N(0,1); x ~ N(0,1)) averages to 0.2337


def test_crps_gaussian_is_the_textbook_formula_and_refuses_nothing():
    v = M.crps_gaussian(np.array([0.0]), np.array([1.0]), np.array([0.0]))
    assert abs(v[0] - (2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))) < 1e-12        # z = 0: 2 phi(0) - 1/sqrt(pi)
    assert M.crps_gaussian(np.array([1.0]), np.array([0.0]), np.array([3.0]))[0] == pytest.approx(2.0, abs=1e-6)   # sigma -> 0: |x - mu|


def test_crps_samples_refuses_fewer_than_two_samples():
    with pytest.raises(ValueError, match='K'):
        M.crps_samples(np.zeros((1, 4)), np.zeros(4))


def test_crps_samples_of_identical_samples_is_the_absolute_error():
    s = np.full((8, 5), 0.25)
    assert np.allclose(M.crps_samples(s, np.full(5, 0.75)), 0.5)


def test_coverage_of_gaussian_samples_is_near_nominal_and_boundaries_count():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(100_000)
    s = rng.standard_normal((32, 100_000))
    cov = M.coverage_samples(s, x)
    assert abs(cov[0.5] - 0.5) < 0.02 and abs(cov[0.9] - 0.9) < 0.02
    const = np.zeros((4, 3))
    assert M.coverage_samples(const, np.zeros(3)) == {0.5: 1.0, 0.9: 1.0}            # x == lo == hi is covered
    assert M.coverage_samples(const, np.ones(3)) == {0.5: 0.0, 0.9: 0.0}
    g = M.coverage_gaussian(np.zeros(100_000), np.ones(100_000), x)
    assert abs(g[0.5] - 0.5) < 0.01 and abs(g[0.9] - 0.9) < 0.01


def test_ause_is_zero_for_the_oracle_and_positive_for_a_random_ranking():
    rng = np.random.default_rng(2)
    err = rng.standard_normal(10_000) * rng.uniform(0.1, 2.0, 10_000)
    assert M.ause(np.abs(err), err) == pytest.approx(0.0, abs=1e-12)
    assert M.ause(rng.uniform(size=10_000), err) > 0.05
    curve = M.sparsification_curve(np.abs(err), err ** 2)
    assert curve.shape == (50,) and curve[0] == pytest.approx(1.0) and np.all(np.diff(curve) <= 1e-12)


def test_spread_ratio():
    assert M.spread_ratio(np.full(10, 4.0), np.full(10, 2.0)) == pytest.approx(1.0)
    assert M.spread_ratio(np.full(10, 1.0), np.full(10, 2.0)) == pytest.approx(0.25)


def blurred(u):
    """3 x 3 box blur with reflect padding along the last two axes."""
    p = np.pad(u, [(0, 0)] * (u.ndim - 2) + [(1, 1), (1, 1)], mode='reflect')
    out = np.zeros_like(u)
    for dy in range(3):
        for dx in range(3):
            out += p[..., dy:dy + u.shape[-2], dx:dx + u.shape[-1]]
    return out / 9


def test_hf_energy_ratio_and_grad_w1_detect_blur():
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(4, 32, 32))
    assert M.hf_energy_ratio(x, x) == 0.0 and M.grad_w1(x, x) == 0.0
    b = blurred(x)
    assert M.hf_energy_ratio(b, x) < -0.5 and M.grad_w1(b, x) > 0.01
    assert M.laplacian(np.ones((2, 5, 5))).max() == pytest.approx(0.0)          # a flat image has no Laplacian
    assert M.sobel_mag(np.ones((5, 5))).max() == pytest.approx(0.0)


def test_laplacian_and_sobel_use_reflect_padding_not_zeros():
    u = np.ones((1, 6, 6)) * 0.7
    assert np.abs(M.laplacian(u)).max() < 1e-12 and np.abs(M.sobel_mag(u)).max() < 1e-12


def test_to01_converts_tensors_and_arrays():
    t = torch.tensor([-1.0, 0.0, 1.0])
    assert np.allclose(M.to01(t), [0.0, 0.5, 1.0]) and M.to01(t).dtype == np.float64
    assert np.allclose(M.to01(np.array([-1.0, 1.0])), [0.0, 1.0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_uq_metrics.py -q`
Expected: `ImportError: cannot import name 'metrics' from 'scripts.uq'`.

- [ ] **Step 3: Implement**

`scripts/uq/metrics.py`:

```python
"""Uncertainty and texture metrics of Phase 2a (spec §5.3-5.4), numpy, per element or per cube.

Inputs are on the [0, 1] display scale (to01 converts the trainer's [-1, 1]); `samples` are [K, ...] with the sample
axis first; everything else broadcasts over the trailing element axes.
"""
import math

import numpy as np
import torch

FRACTIONS = np.arange(0, 1.0, 0.02)                                  # removal fractions of the sparsification curve
Z = {0.5: 0.6744897501960817, 0.9: 1.6448536269514722}             # central-interval half-widths of N(0, 1)


def to01(t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return (a.astype(np.float64) + 1.0) / 2.0


def crps_samples(samples, x):
    """Fair (unbiased) sample CRPS: mean_k |s_k - x| - sum_{j<k} |s_j - s_k| / (K (K - 1)); the pairwise sum is
    sum_i (2 i - K + 1) s_(i) over the ascending order statistics."""
    s = np.sort(np.asarray(samples, dtype=np.float64), axis=0)
    K = s.shape[0]
    if K < 2:
        raise ValueError(f'crps_samples needs K >= 2 samples, got K={K}')
    x = np.asarray(x, dtype=np.float64)
    term1 = np.abs(s - x[None]).mean(0)
    w = (2.0 * np.arange(K) - K + 1.0)
    term2 = np.tensordot(w, s, axes=(0, 0)) / (K * (K - 1))
    return term1 - term2


def crps_gaussian(mu, sigma, x):
    mu, x = np.asarray(mu, dtype=np.float64), np.asarray(x, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    z = (x - mu) / sigma
    Phi = 0.5 * (1.0 + torch.erf(torch.from_numpy(z / math.sqrt(2.0))).numpy())
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def coverage_samples(samples, x, levels=(0.5, 0.9)):
    s, x = np.asarray(samples, dtype=np.float64), np.asarray(x, dtype=np.float64)
    out = {}
    for lv in levels:
        lo, hi = np.quantile(s, [(1.0 - lv) / 2.0, 1.0 - (1.0 - lv) / 2.0], axis=0, method='linear')
        out[lv] = float(np.mean((x >= lo) & (x <= hi)))
    return out


def coverage_gaussian(mu, sigma, x, levels=(0.5, 0.9)):
    mu, sigma, x = (np.asarray(a, dtype=np.float64) for a in (mu, sigma, x))
    return {lv: float(np.mean(np.abs(x - mu) <= Z[lv] * sigma)) for lv in levels}


def sparsification_curve(score, err2, fractions=FRACTIONS):
    """RMSE of the elements left after removing the top-f fraction by `score` (descending), divided by the full RMSE."""
    order = np.argsort(-np.asarray(score, dtype=np.float64).ravel(), kind='stable')
    e = np.asarray(err2, dtype=np.float64).ravel()[order]
    n = e.size
    tail = np.cumsum(e[::-1])[::-1]                                   # tail[i] = sum e[i:]
    full = e.mean()
    out = np.empty(len(fractions))
    for j, f in enumerate(fractions):
        i = int(round(f * n))
        out[j] = math.sqrt(tail[i] / (n - i) / full) if full > 0 else 1.0
    return out


def ause(pred_std, err):
    err2 = np.asarray(err, dtype=np.float64) ** 2
    return float(np.mean(sparsification_curve(pred_std, err2) - sparsification_curve(np.abs(err), err2)))


def spread_ratio(pred_var, err):
    return float(np.mean(pred_var) / np.mean(np.asarray(err, dtype=np.float64) ** 2))


def _pad(u):
    return np.pad(np.asarray(u, dtype=np.float64), [(0, 0)] * (np.ndim(u) - 2) + [(1, 1), (1, 1)], mode='reflect')


def laplacian(u):
    p = _pad(u)
    return p[..., :-2, 1:-1] + p[..., 2:, 1:-1] + p[..., 1:-1, :-2] + p[..., 1:-1, 2:] - 4.0 * np.asarray(u, dtype=np.float64)


def hf_energy_ratio(x_hat, x):
    e_hat, e = float((laplacian(x_hat) ** 2).mean()), float((laplacian(x) ** 2).mean())
    return float(math.log(max(e_hat, 1e-20) / max(e, 1e-20)))


def sobel_mag(u):
    p = _pad(u)
    gx = (p[..., :-2, 2:] + 2.0 * p[..., 1:-1, 2:] + p[..., 2:, 2:]) - (p[..., :-2, :-2] + 2.0 * p[..., 1:-1, :-2] + p[..., 2:, :-2])
    gy = (p[..., 2:, :-2] + 2.0 * p[..., 2:, 1:-1] + p[..., 2:, 2:]) - (p[..., :-2, :-2] + 2.0 * p[..., :-2, 1:-1] + p[..., :-2, 2:])
    return np.sqrt(gx * gx + gy * gy) / 8.0


def grad_w1(x_hat, x, bins=100, vmax=0.5):
    edges = np.linspace(0.0, vmax, bins + 1)
    h1, _ = np.histogram(np.clip(sobel_mag(x_hat), 0, vmax), edges)
    h2, _ = np.histogram(np.clip(sobel_mag(x), 0, vmax), edges)
    c1, c2 = np.cumsum(h1) / max(h1.sum(), 1), np.cumsum(h2) / max(h2.sum(), 1)
    return float(np.abs(c1 - c2).sum() * (vmax / bins))
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_uq_metrics.py -q`
Expected: 10 passed (the Gaussian CRPS test takes a few seconds).

- [ ] **Step 5: Commit**

```bash
git add scripts/uq/metrics.py tests/test_uq_metrics.py
git commit -F <message file>   # "feat(uq): CRPS, coverage, AUSE, spread ratio, HF-energy ratio and gradient-W1 metrics with closed-form tests (Phase 2a Task 7)" + trailers
```

---
### Task 8: The evaluation script `uq_eval.py`

**Files:**
- Create: `scripts/uq_eval.py`
- Test: `tests/test_uq_eval.py`

**Interfaces:**
- Consumes: Task 2 (`compose`, `regress_forward`, `varhead_forward`, `LOGVAR_RANGE`, `build_residual_model`), Task 3 (`meta_sha256`), Task 7 (`metrics as M`), `scripts.eval_warmstart` (`metrics`, `dataset_for`, `load_subset`, `make_eval_loader`, `yswap_batch`, `class_of`), `DiffusionTrainer`, `UQ_DIR`, `file_sha256`, `git_head`.
- Produces:
  - `SAMPLERS = {'ddim10': dict(method='ddim', n_steps=10), 'ddpm1000': dict(method='ddpm', n_steps=1000)}`.
  - `PREREG_CENTRE = {8: ('regress_l1_sel_d8_b64', 11), 4: ('regress_l1_sel_d4_b64', 17)}` (imported by Task 9).
  - `load_ckpt(path) -> (ck, meta)`, `ckpt_epoch(path) -> int`, `ckpt_record(path, meta) -> dict(path, run, epoch, sha256, mode, d, train_half, sigma_d, sigma_r, centre, xr_cache_sha256)`, `expect(meta, path, **want)` (ValueError naming each mismatch).
  - `check_models(args) -> (models: dict, sigma_r: float | None, provenance: dict)`; `models` keys `centre`, `diffusion`, `varhead`, `ensemble` (list), `b`, `halves` (list of 2) — absent ones are `None` / `[]`.
  - `point_metrics(x_hat, x) -> dict(rmse_pct, psnr, sam_deg, hf, w1 -> lists)`; `sample_metrics(samples [K, B, C, H, W], x) -> dict(mean_*, single_*, crps, cov50, cov90, cov50_gauss, cov90_gauss, ause, spread_ratio)`; `gaussian_metrics(mu, sigma, x) -> dict(mean_*, crps, cov50, cov90, ause, spread_ratio)` (sigma on the [−1, 1] scale, divided by 2 inside); CRPS in %.
  - `draw_samples(model, trainer, cond, shape, sigma, cfg, K, centre) -> [K, B, C, H, W]` (seeds 0 … K−1 via `torch.manual_seed(k)` before each draw; `clamp_x0 = (−2/σ, 2/σ)`).
  - `run_eval(models, sigma_r, ds, loader, ys, args, provenance, device) -> record`; `ys = (files, x, y, xh)` the y-swap batch tensors; `write_record(path, record)` (refuses overwrite); `output_path(d, name) -> UQ_DIR/d<d>/eval/<name>.json`.
  - Record schema: `name, d, K, sampler, n, files, classes, sigma_d, sigma_r, per_cube{method{metric: [n]}}, means{method{metric: float}}, yswap{K, files, true: [m], swap_full: [m]} | None, stored_cubes (path | None), provenance{centre, diffusion, varhead, ensemble, b, halves, xr_cache, git_head, created, seeds, device, batch_size, subset, limit_files}`.
  - CLI: `python scripts/uq_eval.py --d 8 --name <name> --centre CKPT [--diffusion CKPT] [--varhead CKPT] [--ensemble CKPT ...] [--b CKPT] [--centre_halves CKPT CKPT] [--xr_cache DIR] [--K 32] [--sampler ddim10|ddpm1000] [--subset gate|ddpm] [--device cuda:0] [--batch_size 4] [--num_workers 4] [--store_cubes 8] [--limit_files 0] [--yswap_n 32]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_uq_eval.py`:

```python
"""Phase 2a Task 8: uq_eval scores every method per cube into one record, refuses mismatched checkpoints and refuses
to overwrite; stub models and a stub dataset on CPU."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model.diffusion_trainer import DiffusionTrainer
from scripts import uq_eval as UE

ROOT = '/fake/root'


class StubDS(torch.utils.data.Dataset):
    sigma_d = 0.1
    d = 4

    def __init__(self, n=6, H=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.img_list = [os.path.join(ROOT, f'MatFlower60/Train/P02{2 + i % 3}/{i}_1.mat') for i in range(n)]
        self.xh = torch.rand(n, H, H, 64, generator=g) * 1.2 - 0.6
        self.x = (self.xh + 0.1 * torch.randn(n, H, H, 64, generator=g)).clamp(-1, 1)
        self.y = torch.rand(n, H // 4, H // 4, 30, generator=g)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        return self.x[i].numpy(), self.y[i].numpy(), self.xh[i].numpy()


class Const(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x_t, cond, t):
        return torch.full_like(x_t, self.c)


class Noisy(torch.nn.Module):
    """A v-prediction net that depends on x_t, so DDIM draws differ across seeds."""
    def forward(self, x_t, cond, t):
        return 0.1 * x_t


def fake_args(tmp_path, **kw):
    a = SimpleNamespace(d=4, name='t', K=3, sampler='ddim10', subset='gate', device='cpu', batch_size=4, num_workers=0,
                        store_cubes=2, limit_files=0, yswap_n=4, xr_cache='', centre='c', diffusion='d', varhead='v',
                        ensemble=['e1', 'e2'], b=None, centre_halves=None, data_root=ROOT, prior='p')
    a.__dict__.update(kw)
    return a


def test_point_and_sample_and_gaussian_metrics_shapes():
    x = torch.rand(2, 64, 8, 8) * 2 - 1
    pm = UE.point_metrics(x, x)
    assert set(pm) == {'rmse_pct', 'psnr', 'sam_deg', 'hf', 'w1'} and pm['rmse_pct'] == [0.0, 0.0] and pm['hf'] == [0.0, 0.0]
    s = x[None] + 0.02 * torch.randn(5, 2, 64, 8, 8)
    sm = UE.sample_metrics(s, x)
    assert len(sm['crps']) == 2 and all(0 < v < 5 for v in sm['crps']) and all(0 <= v <= 1 for v in sm['cov90'])
    assert set(sm) >= {'mean_rmse_pct', 'single_rmse_pct', 'mean_hf', 'single_hf', 'mean_w1', 'single_w1', 'crps', 'cov50', 'cov90',
                       'cov50_gauss', 'cov90_gauss', 'ause', 'spread_ratio'}
    gm = UE.gaussian_metrics(x, torch.full_like(x, 0.02), x)
    assert gm['mean_rmse_pct'] == [0.0, 0.0] and all(v > 0 for v in gm['crps']) and gm['cov90'] == [1.0, 1.0]


def test_draw_samples_seeds_are_reproducible_and_differ_across_k():
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='v', snr_gamma=None)
    centre = torch.zeros(2, 64, 8, 8)
    cond = {'sensor': torch.zeros(2, 30, 2, 2), 'x0_hat': centre}
    s1 = UE.draw_samples(Noisy(), tr, cond, tuple(centre.shape), 0.1, UE.SAMPLERS['ddim10'], 3, centre)
    s2 = UE.draw_samples(Noisy(), tr, cond, tuple(centre.shape), 0.1, UE.SAMPLERS['ddim10'], 3, centre)
    assert s1.shape == (3, 2, 64, 8, 8) and torch.equal(s1, s2) and not torch.equal(s1[0], s1[1])
    assert s1.abs().max() <= 1.0


def test_run_eval_writes_every_method_and_the_yswap_block(tmp_path):
    ds = StubDS()
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=UE.residual_collate_fn)
    files = ds.img_list[:4]
    ys = (files, *UE.residual_collate_fn([ds[i] for i in range(4)]))
    models = dict(centre=Const(0.0), diffusion=Noisy(), varhead=Const(0.0), ensemble=[Const(0.1), Const(-0.1)], b=Noisy(), halves=None)
    prov = dict(centre={'run': 'c'}, diffusion={'run': 'd'}, varhead=None, ensemble=[], b=None, halves=None, xr_cache=None)
    args = fake_args(tmp_path)
    rec = UE.run_eval(models, 0.1, ds, loader, ys, args, prov, torch.device('cpu'), store_dir=str(tmp_path))
    assert rec['n'] == 6 and rec['K'] == 3 and rec['d'] == 4 and rec['sigma_r'] == 0.1 and rec['sigma_d'] == 0.1
    assert set(rec['per_cube']) == {'baseline', 'regressor', 'diffusion', 'varhead', 'ensemble', 'b'}
    for m, d in rec['per_cube'].items():
        for k, v in d.items():
            assert len(v) == 6, (m, k)
    assert rec['per_cube']['regressor']['rmse_pct'] == rec['per_cube']['baseline']['rmse_pct']      # centre = zero residual
    assert rec['means']['diffusion']['crps'] > 0 and 'cov90' in rec['means']['varhead'] and 'crps' in rec['means']['ensemble']
    assert rec['yswap']['K'] == 3 and len(rec['yswap']['true']) == 4 == len(rec['yswap']['swap_full']) and rec['yswap']['files'] == [os.path.relpath(f, ROOT) for f in files]
    assert rec['files'] == [os.path.relpath(f, ROOT) for f in ds.img_list] and rec['classes'] == ['P022', 'P023', 'P024'] * 2
    assert rec['provenance']['seeds'] == [0, 1, 2] and rec['stored_cubes'].endswith('__cubes.npz') and os.path.exists(rec['stored_cubes'])
    z = np.load(rec['stored_cubes'])
    assert z['samples'].shape == (3, 2, 64, 8, 8) and z['samples'].dtype == np.float16


def test_run_eval_without_diffusion_or_varhead_is_the_stage_0_dry_run(tmp_path):
    ds = StubDS()
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=UE.residual_collate_fn)
    models = dict(centre=Const(0.0), diffusion=None, varhead=None, ensemble=[Const(0.1), Const(-0.1)], b=Noisy(), halves=None)
    prov = dict(centre={'run': 'c'}, diffusion=None, varhead=None, ensemble=[], b={'run': 'b'}, halves=None, xr_cache=None)
    rec = UE.run_eval(models, None, ds, loader, None, fake_args(tmp_path, diffusion=None, varhead=None, K=2), prov, torch.device('cpu'), store_dir=str(tmp_path))
    assert set(rec['per_cube']) == {'baseline', 'regressor', 'ensemble', 'b'} and rec['yswap'] is None and rec['sigma_r'] is None
    assert len(rec['per_cube']['b']['crps']) == 6


def test_write_record_refuses_overwrite(tmp_path):
    p = str(tmp_path / 'uq' / 'd4' / 'eval' / 'x.json')
    UE.write_record(p, dict(a=1))
    assert json.load(open(p)) == dict(a=1)
    with pytest.raises(FileExistsError):
        UE.write_record(p, dict(a=2))
    assert UE.output_path(8, 'run1').endswith('/uq/d8/eval/run1.json')


def test_expect_names_every_mismatch():
    UE.expect(dict(mode='regress', d=8), 'p', mode='regress', d=8)
    with pytest.raises(ValueError) as e:
        UE.expect(dict(mode='residual', d=4), 'p', mode='regress', d=8)
    assert 'mode' in str(e.value) and 'd' in str(e.value) and 'p' in str(e.value)


def test_check_models_ties_the_diffusion_to_its_centre_and_cache(tmp_path, monkeypatch):
    from model.residual_wrapper import build_residual_model
    def save(name, meta):
        p = tmp_path / name
        torch.save(dict(model_state_dict=build_residual_model(meta['mode'], 8).state_dict(), meta=meta), p)
        return str(p)
    centre = save('centre.pth', dict(mode='regress', d=4, base_channels=8, sigma_d=0.1, args={}))
    from data_loader.warmstart_paths import file_sha256
    good = dict(mode='residual_reg', d=4, base_channels=8, sigma_d=0.09, sigma_r=0.09, xr_cache_sha256='cachesha',
                centre=dict(run='centre', epoch=1, sha256=file_sha256(centre)), args={})
    diff = save('diff.pth', good)
    monkeypatch.setattr(UE, 'meta_sha256', lambda d: 'cachesha')
    models, sigma_r, prov = UE.check_models(fake_args(tmp_path, centre=centre, diffusion=diff, varhead=None, ensemble=[], xr_cache='/c'), torch.device('cpu'))
    assert sigma_r == 0.09 and models['diffusion'] is not None and prov['diffusion']['sha256'] == file_sha256(diff) and prov['xr_cache'] == dict(dir='/c', sha256='cachesha')
    bad = save('bad.pth', dict(good, centre=dict(run='other', epoch=1, sha256='nope')))
    with pytest.raises(ValueError, match='centre'):
        UE.check_models(fake_args(tmp_path, centre=centre, diffusion=bad, varhead=None, ensemble=[]), torch.device('cpu'))
    monkeypatch.setattr(UE, 'meta_sha256', lambda d: 'rebuilt')
    with pytest.raises(ValueError, match='xr_cache'):
        UE.check_models(fake_args(tmp_path, centre=centre, diffusion=diff, varhead=None, ensemble=[], xr_cache='/c'), torch.device('cpu'))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_uq_eval.py -q`
Expected: `ImportError: cannot import name 'uq_eval'`.

- [ ] **Step 3: Implement**

`scripts/uq_eval.py`:

```python
"""Phase 2a evaluation (spec §5): K samples per gate cube from the regressor-centred residual diffusion, the variance
head's and the deep ensemble's Gaussian predictions, the regressor and the baseline as point estimates, and (d = 4)
B's samples around x0_hat; CRPS, coverage, AUSE, spread ratio, HF-energy ratio, gradient W1, RMSE/PSNR/SAM per cube;
the y-swap of the mean-of-K; one record per run under uq/d<d>/eval/<name>.json (never overwritten).

    python scripts/uq_eval.py --d 8 --name d8_main --centre runs/regress_l1_sel_d8_b64/checkpoint_epoch_11.pth \
        --diffusion runs/residual_reg_d8_b64/checkpoint_epoch_100.pth --varhead runs/varhead_d8_b64/checkpoint_epoch_E.pth \
        --ensemble <regressor ckpts> --xr_cache $R/uq/d8/xr_cache --K 32 --sampler ddim10 --device cuda:0
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from data_loader.hfd_residual import residual_collate_fn  # noqa: E402
from data_loader.warmstart_paths import DATA_ROOT, PRIOR, UQ_DIR, file_sha256, git_head  # noqa: E402
from data_loader.xr_cache import meta_sha256  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from model.residual_wrapper import LOGVAR_RANGE, build_residual_model, compose, regress_forward, varhead_forward  # noqa: E402
from scripts.eval_warmstart import class_of, dataset_for, load_subset, make_eval_loader, metrics, yswap_batch  # noqa: E402
from scripts.uq import metrics as M  # noqa: E402

SAMPLERS = {'ddim10': dict(method='ddim', n_steps=10), 'ddpm1000': dict(method='ddpm', n_steps=1000)}
PREREG_CENTRE = {8: ('regress_l1_sel_d8_b64', 11), 4: ('regress_l1_sel_d4_b64', 17)}      # spec §4.3; checked by uq_verdict
N_TIMESTEPS = 1000


# ----------------------------------------------------------------------------- checkpoints
def load_ckpt(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck, ((ck.get('meta') if isinstance(ck, dict) else None) or {})


def ckpt_epoch(path):
    m = re.search(r'checkpoint_epoch_(\d+)\.pth$', path)
    return int(m.group(1)) if m else None


def ckpt_record(path, meta):
    return dict(path=os.path.abspath(path), run=os.path.basename(os.path.dirname(os.path.abspath(path))), epoch=ckpt_epoch(path),
                sha256=file_sha256(path), mode=meta.get('mode'), d=meta.get('d'), train_half=int((meta.get('args') or {}).get('train_half', 0) or 0),
                sigma_d=meta.get('sigma_d'), sigma_r=meta.get('sigma_r'), centre=meta.get('centre'), xr_cache_sha256=meta.get('xr_cache_sha256'))


def expect(meta, path, **want):
    bad = [f'{k}: checkpoint {meta.get(k, "<missing>")!r} vs required {v!r}' for k, v in want.items() if meta.get(k) != v]
    if bad:
        raise ValueError(f'checkpoint {path} does not fit this evaluation: ' + '; '.join(bad))


def _load(path, meta, device):
    ck, _ = load_ckpt(path)
    model = build_residual_model(meta['mode'], int(meta['base_channels']))
    model.load_state_dict(ck['model_state_dict'], strict=True)
    return model.to(device).eval()


def check_models(args, device):
    models = dict(centre=None, diffusion=None, varhead=None, ensemble=[], b=None, halves=None)
    prov = dict(centre=None, diffusion=None, varhead=None, ensemble=[], b=None, halves=None, xr_cache=None)
    _, mc = load_ckpt(args.centre)
    expect(mc, args.centre, mode='regress', d=args.d)
    models['centre'], prov['centre'] = _load(args.centre, mc, device), ckpt_record(args.centre, mc)
    centre_sha, sigma_r = prov['centre']['sha256'], None
    cache_sha = meta_sha256(args.xr_cache) if getattr(args, 'xr_cache', '') else None
    if cache_sha is not None:
        prov['xr_cache'] = dict(dir=args.xr_cache, sha256=cache_sha)
    for key, mode in (('diffusion', 'residual_reg'), ('varhead', 'varhead')):
        path = getattr(args, key, None)
        if not path:
            continue
        _, meta = load_ckpt(path)
        expect(meta, path, mode=mode, d=args.d)
        if (meta.get('centre') or {}).get('sha256') != centre_sha:
            raise ValueError(f'{key} checkpoint {path} was trained around centre {meta.get("centre")!r} but --centre is '
                             f'{prov["centre"]["run"]}@{prov["centre"]["epoch"]} ({centre_sha[:12]}...)')
        if cache_sha is not None and meta.get('xr_cache_sha256') != cache_sha:
            raise ValueError(f'{key} checkpoint {path}: xr_cache_sha256 {meta.get("xr_cache_sha256")!r} differs from the cache '
                             f'on disk {cache_sha!r} (the cache was rebuilt)')
        s = float(meta['sigma_r'])
        if sigma_r is not None and abs(s - sigma_r) >= 1e-9:
            raise ValueError(f'{key} sigma_r {s} differs from the diffusion sigma_r {sigma_r}')
        sigma_r = s
        models[key], prov[key] = _load(path, meta, device), ckpt_record(path, meta)
    for path in (getattr(args, 'ensemble', None) or []):
        _, meta = load_ckpt(path)
        expect(meta, path, mode='regress', d=args.d)
        models['ensemble'].append(_load(path, meta, device)); prov['ensemble'].append(ckpt_record(path, meta))
    if getattr(args, 'b', None):
        _, meta = load_ckpt(args.b)
        expect(meta, args.b, mode='residual', d=args.d)
        models['b'], prov['b'] = _load(args.b, meta, device), ckpt_record(args.b, meta)
    if getattr(args, 'centre_halves', None):
        models['halves'], prov['halves'] = [], []
        for path, th in zip(args.centre_halves, (1, 2)):
            _, meta = load_ckpt(path)
            expect(meta, path, mode='regress', d=args.d)
            if int((meta.get('args') or {}).get('train_half', 0) or 0) != th:
                raise ValueError(f'--centre_halves: {path} is not a half-{th} regressor')
            models['halves'].append(_load(path, meta, device)); prov['halves'].append(ckpt_record(path, meta))
    return models, sigma_r, prov


# ----------------------------------------------------------------------------- metrics per batch
def point_metrics(x_hat, x):
    m = {k: v.tolist() for k, v in metrics(x_hat, x).items()}
    a, b = M.to01(x_hat), M.to01(x)
    m['hf'] = [M.hf_energy_ratio(a[i], b[i]) for i in range(a.shape[0])]
    m['w1'] = [M.grad_w1(a[i], b[i]) for i in range(a.shape[0])]
    return m


def sample_metrics(samples, x):
    out = {f'mean_{k}': v for k, v in point_metrics(samples.mean(0), x).items()}
    out.update({f'single_{k}': v for k, v in point_metrics(samples[0], x).items()})
    s01, x01 = M.to01(samples), M.to01(x)
    keys = ('crps', 'cov50', 'cov90', 'cov50_gauss', 'cov90_gauss', 'ause', 'spread_ratio')
    acc = {k: [] for k in keys}
    for i in range(x01.shape[0]):
        s, t = s01[:, i], x01[i]
        mu, sd = s.mean(0), s.std(0, ddof=1)
        c, g = M.coverage_samples(s, t), M.coverage_gaussian(mu, sd, t)
        acc['crps'].append(100.0 * float(M.crps_samples(s, t).mean()))
        acc['cov50'].append(c[0.5]); acc['cov90'].append(c[0.9]); acc['cov50_gauss'].append(g[0.5]); acc['cov90_gauss'].append(g[0.9])
        acc['ause'].append(M.ause(sd, mu - t)); acc['spread_ratio'].append(M.spread_ratio(sd ** 2, mu - t))
    out.update(acc)
    return out


def gaussian_metrics(mu, sigma, x):
    out = {f'mean_{k}': v for k, v in point_metrics(mu, x).items()}
    mu01, x01 = M.to01(mu), M.to01(x)
    sd01 = sigma.detach().double().cpu().numpy() / 2.0
    acc = {k: [] for k in ('crps', 'cov50', 'cov90', 'ause', 'spread_ratio')}
    for i in range(x01.shape[0]):
        m_, s_, t = mu01[i], sd01[i], x01[i]
        c = M.coverage_gaussian(m_, s_, t)
        acc['crps'].append(100.0 * float(M.crps_gaussian(m_, s_, t).mean()))
        acc['cov50'].append(c[0.5]); acc['cov90'].append(c[0.9])
        acc['ause'].append(M.ause(s_, m_ - t)); acc['spread_ratio'].append(M.spread_ratio(s_ ** 2, m_ - t))
    out.update(acc)
    return out


@torch.no_grad()
def draw_samples(model, trainer, cond, shape, sigma, cfg, K, centre):
    out = []
    for k in range(K):
        torch.manual_seed(k)
        r = trainer.sample(model, cond, shape, n_steps=cfg['n_steps'], method=cfg['method'], progress=False,
                           clamp_x0=(-2.0 / sigma, 2.0 / sigma))
        out.append(compose(centre, sigma, r))
    return torch.stack(out, 0)


@torch.no_grad()
def regress_centre(model, y, xh, sigma_d):
    return compose(xh, sigma_d, regress_forward(model, {'sensor': y, 'x0_hat': xh}))


# ----------------------------------------------------------------------------- the run
def _extend(acc, method, d):
    a = acc.setdefault(method, {})
    for k, v in d.items():
        a.setdefault(k, []).extend(v)


def run_eval(models, sigma_r, ds, loader, ys, args, provenance, device, store_dir=None):
    cfg = SAMPLERS[args.sampler]
    K, sigma_d = int(args.K), float(ds.sigma_d)
    trainer = DiffusionTrainer(device=device, n_timesteps=N_TIMESTEPS, prediction_type='v', loss_type='l1', snr_gamma=None)
    acc, stored, t0 = {}, None, time.time()
    for bi, (x, y, xh) in enumerate(loader):
        x, y, xh = x.to(device), y.to(device), xh.to(device)
        xr = regress_centre(models['centre'], y, xh, sigma_d)
        cond = {'sensor': y, 'x0_hat': xr}
        _extend(acc, 'baseline', point_metrics(xh, x))
        _extend(acc, 'regressor', point_metrics(xr, x))
        if models['diffusion'] is not None:
            s = draw_samples(models['diffusion'], trainer, cond, tuple(xr.shape), sigma_r, cfg, K, xr)
            _extend(acc, 'diffusion', sample_metrics(s, x))
            if bi == 0 and store_dir is not None and args.store_cubes:
                n = min(int(args.store_cubes), x.shape[0])
                stored = os.path.join(store_dir, f'{args.name}__cubes.npz')
                np.savez(stored, samples=s[:, :n].half().cpu().numpy(), x=x[:n].cpu().numpy(), x0_hat=xh[:n].cpu().numpy(),
                         x_r=xr[:n].cpu().numpy(), files=np.array([os.path.relpath(f, args.data_root) for f in ds.img_list[:n]]))
        if models['varhead'] is not None:
            logvar = varhead_forward(models['varhead'], cond).clamp(*LOGVAR_RANGE)
            _extend(acc, 'varhead', gaussian_metrics(xr, sigma_r * torch.exp(0.5 * logvar), x))
        if models['ensemble']:
            members = torch.stack([regress_centre(m, y, xh, sigma_d) for m in models['ensemble']], 0)
            _extend(acc, 'ensemble', gaussian_metrics(members.mean(0), members.std(0, unbiased=True), x))
        if models['b'] is not None:
            sb = draw_samples(models['b'], trainer, {'sensor': y, 'x0_hat': xh}, tuple(xh.shape), sigma_d, cfg, K, xh)
            _extend(acc, 'b', sample_metrics(sb, x))
        if models['halves'] and models['diffusion'] is not None:
            xr_h = torch.stack([regress_centre(m, y, xh, sigma_d) for m in models['halves']], 0).mean(0)
            _extend(acc, 'regressor_halves', point_metrics(xr_h, x))
            sh = draw_samples(models['diffusion'], trainer, {'sensor': y, 'x0_hat': xr_h}, tuple(xr.shape), sigma_r, cfg, K, xr_h)
            _extend(acc, 'diffusion_halfcentre', sample_metrics(sh, x))
        print(f'batch {bi + 1}/{len(loader)} ({time.time() - t0:.0f} s)', flush=True)
    yswap = None
    if models['diffusion'] is not None and ys is not None:
        files, x, y, xh = ys
        x, y, xh = x.to(device), y.to(device), xh.to(device)
        xr = regress_centre(models['centre'], y, xh, sigma_d)
        k = x.shape[0] // 2
        s_true = draw_samples(models['diffusion'], trainer, {'sensor': y, 'x0_hat': xr}, tuple(xr.shape), sigma_r, cfg, K, xr).mean(0)
        s_swap = draw_samples(models['diffusion'], trainer, {'sensor': y.roll(k, 0), 'x0_hat': xr.roll(k, 0)}, tuple(xr.shape), sigma_r, cfg, K, xr).mean(0)
        yswap = dict(K=K, files=[os.path.relpath(f, args.data_root) for f in files], true=metrics(s_true, x)['rmse_pct'].tolist(),
                     swap_full=metrics(s_swap, x)['rmse_pct'].tolist(), baseline=metrics(xh, x)['rmse_pct'].tolist(),
                     regressor=metrics(xr, x)['rmse_pct'].tolist())
    means = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items()}
    return dict(name=args.name, d=int(args.d), K=K, sampler=args.sampler, n=len(ds), files=[os.path.relpath(f, args.data_root) for f in ds.img_list],
                classes=[class_of(f) for f in ds.img_list], sigma_d=sigma_d, sigma_r=sigma_r, per_cube=acc, means=means, yswap=yswap,
                stored_cubes=stored, provenance=dict(provenance, git_head=git_head(), created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                                                      seeds=list(range(K)), device=str(device), batch_size=int(args.batch_size),
                                                      subset=args.subset, limit_files=int(args.limit_files), seconds=time.time() - t0))


def output_path(d, name):
    return os.path.join(UQ_DIR, f'd{d}', 'eval', f'{name}.json')


def write_record(path, record):
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; records are never overwritten (pick another --name)')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(record, f)
    os.replace(tmp, path)


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--centre', required=True); ap.add_argument('--diffusion', default=None); ap.add_argument('--varhead', default=None)
    ap.add_argument('--ensemble', nargs='*', default=[]); ap.add_argument('--b', default=None); ap.add_argument('--centre_halves', nargs=2, default=None)
    ap.add_argument('--xr_cache', default='')
    ap.add_argument('--K', type=int, default=32); ap.add_argument('--sampler', choices=sorted(SAMPLERS), default='ddim10')
    ap.add_argument('--subset', choices=['gate', 'ddpm'], default='gate'); ap.add_argument('--yswap_n', type=int, default=32)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size', type=int, default=4); ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--store_cubes', type=int, default=8); ap.add_argument('--limit_files', type=int, default=0)
    ap.add_argument('--data_root', default=DATA_ROOT); ap.add_argument('--prior', default=PRIOR)
    return ap


def main():
    args = get_parser().parse_args()
    out = output_path(args.d, args.name)
    if os.path.exists(out):
        raise FileExistsError(f'{out} exists; pick another --name')
    device = torch.device(args.device)
    models, sigma_r, prov = check_models(args, device)
    files = load_subset(args.data_root, 'files' if args.subset == 'gate' else 'ddpm1000_subset')
    if args.limit_files:
        files = files[:args.limit_files]
    ds = dataset_for(args.data_root, args.prior, args.d, files)
    loader = make_eval_loader(ds, args.batch_size, args.num_workers)
    ys = None
    if models['diffusion'] is not None:
        ds_ys, batch = yswap_batch(args.data_root, args.prior, args.d, n=args.yswap_n)
        ys = (list(ds_ys.img_list), *batch)
    rec = run_eval(models, sigma_r, ds, loader, ys, args, prov, device, store_dir=os.path.dirname(out))
    write_record(out, rec)
    print('wrote', out)
    for m, d in rec['means'].items():
        print(f"  {m:22s} " + ' '.join(f'{k}={v:.4f}' for k, v in d.items() if k in ('rmse_pct', 'mean_rmse_pct', 'single_rmse_pct', 'crps', 'cov90', 'ause', 'mean_hf', 'single_hf', 'hf')))


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_uq_eval.py -q`
Expected: 7 passed (the Noisy-model DDIM draws on 8 × 8 cubes take seconds).

- [ ] **Step 5: Commit**

```bash
git add scripts/uq_eval.py tests/test_uq_eval.py
git commit -F <message file>   # "feat(uq): uq_eval — K-sample scoring of the centred diffusion, variance head, ensemble, B and the point estimates into one record (Phase 2a Task 8)" + trailers
```

---
### Task 9: The verdict script `uq_verdict.py`

**Files:**
- Create: `scripts/uq_verdict.py`
- Test: `tests/test_uq_verdict.py`

**Interfaces:**
- Consumes: Task 8 record schema and `PREREG_CENTRE`; `scripts.paired_stats.cluster_se`; `UQ_DIR`.
- Produces: `paired(a, b, files) -> dict(mean, se, se_cluster, n, clusters)`; `clauses(rec) -> dict` with keys `C1, C2, P1, P2, S1, S2` (each `dict(..., rule=str, passed=bool)`), `reported` (C3 AUSE paired diff, `spread_ratio` means, `cov_gauss`, ensemble paired CRPS/AUSE when present, `diffusion_halfcentre` CRPS when present), `calibration_pass`, `perceptual_pass`, `wins` (= all four); `verdict_path(rec, no_centre_check, report_only) -> UQ_DIR/d<d>/verdict__<name>[__nocentre][__report].json`; `check_centre(rec)` (ValueError unless `rec['provenance']['centre']` run/epoch equal `PREREG_CENTRE[d]`); CLI `python scripts/uq_verdict.py RECORD.json [--no_centre_check] [--report_only]`.
- Thresholds (spec §5.7, verbatim): C1 `mean(CRPS_diff − CRPS_varhead) + 2 SE_cluster < 0`; C2 `0.85 ≤ cov90 ≤ 0.95 and 0.45 ≤ cov50 ≤ 0.55` (empirical, means over cubes); P1 `mean(|hf_single| − |hf_regressor|) + 2 SE_cluster < 0`; P2 `mean(RMSE_single − RMSE_baseline) + 2 SE_cluster < 0`; S1 `|mean(RMSE_meanK − RMSE_regressor)| ≤ 2 SE_cluster`; S2 `mean(RMSE_swap_full − RMSE_true) − 2 SE_iid > 0` on the y-swap cubes.

- [ ] **Step 1: Write the failing tests**

`tests/test_uq_verdict.py`:

```python
"""Phase 2a Task 9: the six pre-registered clauses, each flipped by its own input; the centre check; no overwrite."""
import json
import os

import numpy as np
import pytest

from scripts import uq_verdict as V

N = 40
FILES = [f'MatFlower60/Train/P02{2 + (i // 4) % 3}/{i // 4}_{i % 4}.mat' for i in range(N)]    # 10 images x 4 patches


def record(tmp_path=None, **over):
    rng = np.random.default_rng(0)
    base = 4.5 + rng.normal(0, 0.3, N)
    reg = base - 0.35 + rng.normal(0, 0.05, N)
    pc = dict(
        baseline=dict(rmse_pct=base.tolist(), hf=(-0.9 + rng.normal(0, 0.05, N)).tolist()),
        regressor=dict(rmse_pct=reg.tolist(), hf=(-0.8 + rng.normal(0, 0.05, N)).tolist()),
        diffusion=dict(crps=(1.0 + rng.normal(0, 0.05, N)).tolist(), cov90=(0.90 + rng.normal(0, 0.01, N)).tolist(),
                       cov50=(0.50 + rng.normal(0, 0.01, N)).tolist(), single_hf=(-0.2 + rng.normal(0, 0.05, N)).tolist(),
                       single_rmse_pct=(reg + 0.05 + rng.normal(0, 0.05, N)).tolist(), mean_rmse_pct=(reg + rng.normal(0, 0.01, N)).tolist(),
                       ause=(0.05 + rng.normal(0, 0.005, N)).tolist(), spread_ratio=(1.0 + rng.normal(0, 0.05, N)).tolist(),
                       cov90_gauss=(0.9 + rng.normal(0, 0.01, N)).tolist()),
        varhead=dict(crps=(1.3 + rng.normal(0, 0.05, N)).tolist(), ause=(0.08 + rng.normal(0, 0.005, N)).tolist(),
                     cov90=(0.8 + rng.normal(0, 0.01, N)).tolist(), spread_ratio=(0.7 + rng.normal(0, 0.05, N)).tolist()))
    ys = dict(K=32, files=FILES[:8], true=(4.0 + rng.normal(0, 0.1, 8)).tolist(), swap_full=(4.8 + rng.normal(0, 0.1, 8)).tolist())
    rec = dict(name='t', d=8, K=32, sampler='ddim10', n=N, files=FILES, per_cube=pc, yswap=ys,
               provenance=dict(centre=dict(run='regress_l1_sel_d8_b64', epoch=11)))
    for k, v in over.items():
        rec['per_cube'][k[0]][k[1]] = v if isinstance(k, tuple) else v
    return rec


def test_everything_passes_on_the_reference_record():
    c = V.clauses(record())
    assert all(c[k]['passed'] for k in ('C1', 'C2', 'P1', 'P2', 'S1', 'S2'))
    assert c['calibration_pass'] and c['perceptual_pass'] and c['wins']
    assert c['C1']['clusters'] == 10 and c['C1']['se_cluster'] > 0 and 'C3' in c['reported']


def test_each_clause_fails_on_its_own_input():
    r = record(); r['per_cube']['varhead']['crps'] = (np.array(r['per_cube']['diffusion']['crps']) - 0.01).tolist()
    assert not V.clauses(r)['C1']['passed'] and V.clauses(r)['C2']['passed']
    r = record(); r['per_cube']['diffusion']['cov90'] = [0.8] * N
    assert not V.clauses(r)['C2']['passed'] and not V.clauses(r)['calibration_pass']
    r = record(); r['per_cube']['diffusion']['single_hf'] = r['per_cube']['regressor']['hf']
    assert not V.clauses(r)['P1']['passed'] and not V.clauses(r)['perceptual_pass']
    r = record(); r['per_cube']['diffusion']['single_rmse_pct'] = (np.array(r['per_cube']['baseline']['rmse_pct']) + 0.1).tolist()
    assert not V.clauses(r)['P2']['passed']
    r = record(); r['per_cube']['diffusion']['mean_rmse_pct'] = (np.array(r['per_cube']['regressor']['rmse_pct']) + 0.3).tolist()
    assert not V.clauses(r)['S1']['passed'] and not V.clauses(r)['wins']
    r = record(); r['yswap']['swap_full'] = r['yswap']['true']
    assert not V.clauses(r)['S2']['passed']


def test_clauses_need_every_method():
    r = record(); del r['per_cube']['varhead']
    with pytest.raises(ValueError, match='varhead'):
        V.clauses(r)


def test_centre_check_and_paths():
    V.check_centre(record())
    r = record(); r['provenance']['centre']['epoch'] = 100
    with pytest.raises(ValueError, match='pre-registered'):
        V.check_centre(r)
    assert V.verdict_path(record(), False, False).endswith('/uq/d8/verdict__t.json')
    assert V.verdict_path(record(), True, True).endswith('/uq/d8/verdict__t__nocentre__report.json')


def test_main_writes_once_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(V, 'UQ_DIR', str(tmp_path))
    p = tmp_path / 'rec.json'
    p.write_text(json.dumps(record()))
    V.main([str(p)])
    out = tmp_path / 'd8' / 'verdict__t.json'
    v = json.load(open(out))
    assert v['wins'] is True and v['record'] == str(p) and v['report_only'] is False
    with pytest.raises(FileExistsError):
        V.main([str(p)])
    r = record(); r['provenance']['centre']['epoch'] = 3
    q = tmp_path / 'rec2.json'; q.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        V.main([str(q)])
    V.main([str(q), '--no_centre_check', '--report_only'])
    assert (tmp_path / 'd8' / 'verdict__t__nocentre__report.json').exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_uq_verdict.py -q`
Expected: `ImportError: cannot import name 'uq_verdict'`.

- [ ] **Step 3: Implement**

`scripts/uq_verdict.py`:

```python
"""The pre-registered Phase 2a verdict (spec §5.7) read from one uq_eval record. Written before any Stage 1 number
exists; the thresholds live here and nowhere else.

    python scripts/uq_verdict.py $R/uq/d8/eval/d8_main.json [--no_centre_check] [--report_only]

Calibration passes = C1 and C2; perceptual passes = P1 and P2; "diffusion wins (a)" = both and S1 and S2. The d = 4
contrast runs with --report_only (same clauses, no gate). The record's centre must be the pre-registered selected
regressor of its stride unless --no_centre_check (smoke tests); either flag is spelled into the verdict file name.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from data_loader.warmstart_paths import UQ_DIR, git_head  # noqa: E402
from scripts.paired_stats import cluster_se  # noqa: E402
from scripts.uq_eval import PREREG_CENTRE  # noqa: E402

COV90, COV50 = (0.85, 0.95), (0.45, 0.55)
NEEDED = ('diffusion', 'varhead', 'regressor', 'baseline')


def paired(a, b, files):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    se_c, G = cluster_se(d, files)
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / math.sqrt(len(d))), se_cluster=se_c, n=int(len(d)), clusters=G)


def _m(v):
    return float(np.mean(v))


def clauses(rec):
    pc, files = rec['per_cube'], rec['files']
    missing = [m for m in NEEDED if m not in pc]
    if missing:
        raise ValueError(f'record {rec.get("name")} lacks the methods {missing}; the verdict needs {NEEDED}')
    out = {}
    c1 = paired(pc['diffusion']['crps'], pc['varhead']['crps'], files)
    out['C1'] = dict(c1, rule='mean(CRPS_diffusion - CRPS_varhead) + 2 SE_cluster < 0', passed=bool(c1['mean'] + 2 * c1['se_cluster'] < 0))
    cov90, cov50 = _m(pc['diffusion']['cov90']), _m(pc['diffusion']['cov50'])
    out['C2'] = dict(cov90=cov90, cov50=cov50, rule=f'{COV90[0]} <= cov90 <= {COV90[1]} and {COV50[0]} <= cov50 <= {COV50[1]}',
                     passed=bool(COV90[0] <= cov90 <= COV90[1] and COV50[0] <= cov50 <= COV50[1]))
    p1 = paired(np.abs(pc['diffusion']['single_hf']), np.abs(pc['regressor']['hf']), files)
    out['P1'] = dict(p1, rule='mean(|hf_single| - |hf_regressor|) + 2 SE_cluster < 0', passed=bool(p1['mean'] + 2 * p1['se_cluster'] < 0))
    p2 = paired(pc['diffusion']['single_rmse_pct'], pc['baseline']['rmse_pct'], files)
    out['P2'] = dict(p2, rule='mean(RMSE_single - RMSE_baseline) + 2 SE_cluster < 0', passed=bool(p2['mean'] + 2 * p2['se_cluster'] < 0))
    s1 = paired(pc['diffusion']['mean_rmse_pct'], pc['regressor']['rmse_pct'], files)
    out['S1'] = dict(s1, rule='|mean(RMSE_meanK - RMSE_regressor)| <= 2 SE_cluster', passed=bool(abs(s1['mean']) <= 2 * s1['se_cluster']))
    ys = rec.get('yswap') or {}
    if not ys:
        raise ValueError('record has no yswap block (S2 needs the mean-of-K y-swap)')
    d = np.asarray(ys['swap_full'], dtype=np.float64) - np.asarray(ys['true'], dtype=np.float64)
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    out['S2'] = dict(mean=float(d.mean()), se=se, n=int(len(d)), rule='mean(RMSE_swap_full - RMSE_true) - 2 SE_iid > 0 on the y-swap cubes',
                     passed=bool(d.mean() - 2 * se > 0))
    rep = dict(C3_ause=paired(pc['diffusion']['ause'], pc['varhead']['ause'], files),
               spread_ratio=dict(diffusion=_m(pc['diffusion']['spread_ratio']), varhead=_m(pc['varhead']['spread_ratio'])),
               cov90_varhead=_m(pc['varhead']['cov90']), cov90_gauss_diffusion=_m(pc['diffusion'].get('cov90_gauss', [float('nan')])))
    if 'ensemble' in pc:
        rep['ensemble'] = dict(crps=paired(pc['diffusion']['crps'], pc['ensemble']['crps'], files), ause=paired(pc['diffusion']['ause'], pc['ensemble']['ause'], files),
                               cov90=_m(pc['ensemble']['cov90']))
    if 'diffusion_halfcentre' in pc:
        rep['halfcentre'] = dict(crps=paired(pc['diffusion_halfcentre']['crps'], pc['diffusion']['crps'], files),
                                 cov90=_m(pc['diffusion_halfcentre']['cov90']))
    out['reported'] = rep
    out['calibration_pass'] = bool(out['C1']['passed'] and out['C2']['passed'])
    out['perceptual_pass'] = bool(out['P1']['passed'] and out['P2']['passed'])
    out['wins'] = bool(out['calibration_pass'] and out['perceptual_pass'] and out['S1']['passed'] and out['S2']['passed'])
    return out


def check_centre(rec):
    want = PREREG_CENTRE.get(int(rec['d']))
    c = (rec.get('provenance') or {}).get('centre') or {}
    got = (c.get('run'), c.get('epoch'))
    if want is None or got != want:
        raise ValueError(f'record centre {got} is not the pre-registered selected regressor {want} for d = {rec["d"]} '
                         f'(use --no_centre_check only for smoke tests)')


def verdict_path(rec, no_centre_check, report_only):
    return os.path.join(UQ_DIR, f'd{rec["d"]}', f'verdict__{rec["name"]}' + ('__nocentre' if no_centre_check else '')
                        + ('__report' if report_only else '') + '.json')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('record'); ap.add_argument('--no_centre_check', action='store_true'); ap.add_argument('--report_only', action='store_true')
    args = ap.parse_args(argv)
    with open(args.record) as f:
        rec = json.load(f)
    out = verdict_path(rec, args.no_centre_check, args.report_only)
    if os.path.exists(out):
        raise FileExistsError(f'{out} exists; the verdict is taken once')
    if not args.no_centre_check:
        check_centre(rec)
    c = clauses(rec)
    for k in ('C1', 'C2', 'P1', 'P2', 'S1', 'S2'):
        v = c[k]
        stat = f"{v['mean']:+.4f} ± {v['se']:.4f}" + (f" (cluster {v['se_cluster']:.4f})" if 'se_cluster' in v else '') if 'mean' in v else f"cov90 {v['cov90']:.3f} cov50 {v['cov50']:.3f}"
        print(f"{k}: {'PASS' if v['passed'] else 'FAIL'}  {stat}   [{v['rule']}]")
    print(f"calibration {'PASS' if c['calibration_pass'] else 'FAIL'}; perceptual {'PASS' if c['perceptual_pass'] else 'FAIL'}; "
          f"diffusion wins (a): {c['wins']}" + ('   (REPORT ONLY, no gate)' if args.report_only else ''))
    v = dict(c, record=os.path.abspath(args.record), name=rec['name'], d=rec['d'], K=rec['K'], sampler=rec['sampler'], n=rec['n'],
             centre=(rec.get('provenance') or {}).get('centre'), report_only=bool(args.report_only), no_centre_check=bool(args.no_centre_check),
             git_head=git_head(), created=time.strftime('%Y-%m-%dT%H:%M:%S'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + '.tmp', 'w') as f:
        json.dump(v, f, indent=1)
    os.replace(out + '.tmp', out)
    print('wrote', out)
    return v


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_uq_verdict.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/uq_verdict.py tests/test_uq_verdict.py
git commit -F <message file>   # "feat(uq): uq_verdict — the pre-registered C1/C2/P1/P2/S1/S2 clauses, centre check, one verdict file per record (Phase 2a Task 9)" + trailers
```

---

### Task 10: CPU smoke script and the spec's as-built note

**Files:**
- Create: `scripts/pilot/smoke_uq.sh`
- Modify: `docs/superpowers/specs/2026-10-22-regressor-centred-uq-diffusion-design.md` (append "As built" section)

**Interfaces:**
- Consumes: every script of Tasks 1–9 through their CLIs.
- Produces: `bash scripts/pilot/smoke_uq.sh` exits 0 and prints `uq smoke passed` after a full CPU pass on 16 training files (needs the dataset and the real prior).

- [ ] **Step 1: Write the script**

`scripts/pilot/smoke_uq.sh`:

```bash
#!/bin/bash
# CPU end-to-end smoke of Phase 2a on 16 training files (base 8, 1 epoch each): split -> half regressors -> full
# regressor -> centre cache -> residual_reg -> varhead -> uq_eval (K = 2) -> uq_verdict (--no_centre_check).
# Needs the HFD dataset and the real prior; writes only under $SMOKE_UQ_TMP (default /tmp/smoke_uq_<pid>). ~10 min.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PILOT_PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python}
REAL_ROOT=${REAL_RESULTS_ROOT:-/data/chaoyi_he/HSI/Diffu/results/residual_warmstart}
T=${SMOKE_UQ_TMP:-/tmp/smoke_uq_$$}
rm -rf "$T"; mkdir -p "$T/root/runs" "$T/root/uq"
cp "$REAL_ROOT/linear_prior_R1.npz" "$T/root/"                    # the prior with its sigma_d table
export RESIDUAL_WARMSTART_RESULTS=$T/root CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
R=$T/root/runs; SPLIT=$T/root/uq/split_halves.json; CACHE=$T/root/uq/d4/xr_cache
COMMON="--device cpu --base_channels 8 --batch_size 4 --num_workers 2 --num_epochs 1 --save_every 1 --val_every 1 --keep_last 2 --no_bf16 --ds 4"
REG="--mode regress --prediction_type direct --loss_type l1 --split_json $SPLIT"

$PY scripts/uq/split_halves.py --out "$SPLIT"
$PY main_2d_residual_fsdp.py $REG $COMMON --train_half 1 --limit_files 8 --save_dir "$R/regress_l1_half1_d4_b8"
$PY main_2d_residual_fsdp.py $REG $COMMON --train_half 2 --limit_files 8 --save_dir "$R/regress_l1_half2_d4_b8"
$PY main_2d_residual_fsdp.py $REG $COMMON --limit_files 16 --save_dir "$R/regress_l1_full_d4_b8"
for run in regress_l1_half1_d4_b8 regress_l1_half2_d4_b8 regress_l1_full_d4_b8; do
  [[ "$(cat "$R/$run/DONE")" == "1" ]] || { echo "$run did not finish"; exit 1; }
done
$PY scripts/uq/build_xr_cache.py --d 4 --half1 "$R/regress_l1_half1_d4_b8/checkpoint_epoch_1.pth" \
    --half2 "$R/regress_l1_half2_d4_b8/checkpoint_epoch_1.pth" --full "$R/regress_l1_full_d4_b8/checkpoint_epoch_1.pth" \
    --out "$CACHE" --split_json "$SPLIT" --device cpu --batch_size 4 --num_workers 2 --limit_files 16
[[ -f "$CACHE/meta.json" ]] || { echo 'no cache meta'; exit 1; }
$PY main_2d_residual_fsdp.py --mode residual_reg $COMMON --xr_cache "$CACHE" --limit_files 16 --save_dir "$R/residual_reg_d4_b8"
$PY main_2d_residual_fsdp.py --mode varhead --prediction_type direct --loss_type nll $COMMON --xr_cache "$CACHE" --limit_files 16 --save_dir "$R/varhead_d4_b8"
grep -q '"sigma_r"' <<< "$($PY -c "import torch; print(torch.load('$R/residual_reg_d4_b8/checkpoint_epoch_1.pth', map_location='cpu', weights_only=False)['meta'])")" \
    || { echo 'residual_reg meta lacks sigma_r'; exit 1; }
$PY scripts/uq_eval.py --d 4 --name smoke --centre "$R/regress_l1_full_d4_b8/checkpoint_epoch_1.pth" \
    --diffusion "$R/residual_reg_d4_b8/checkpoint_epoch_1.pth" --varhead "$R/varhead_d4_b8/checkpoint_epoch_1.pth" \
    --ensemble "$R/regress_l1_half1_d4_b8/checkpoint_epoch_1.pth" "$R/regress_l1_half2_d4_b8/checkpoint_epoch_1.pth" \
    --xr_cache "$CACHE" --K 2 --limit_files 4 --yswap_n 4 --device cpu --batch_size 2 --num_workers 0 --store_cubes 2
REC=$T/root/uq/d4/eval/smoke.json
[[ -f "$REC" && -f "$T/root/uq/d4/eval/smoke__cubes.npz" ]] || { echo 'no record'; exit 1; }
$PY scripts/uq_verdict.py "$REC" --no_centre_check --report_only
grep -q '"S2"' "$T/root/uq/d4/verdict__smoke__nocentre__report.json" || { echo 'verdict lacks S2'; exit 1; }
if $PY scripts/uq_verdict.py "$REC" --report_only 2>/dev/null; then echo 'verdict accepted a non-pre-registered centre'; exit 1; fi
echo "uq smoke passed ($T)"
```

- [ ] **Step 2: Run it**

Run: `bash scripts/pilot/smoke_uq.sh`
Expected: `uq smoke passed (/tmp/smoke_uq_<pid>)`. Fix whatever breaks in the task that owns it (with a test), not in the smoke script.

- [ ] **Step 3: Append the as-built note to the spec**

Append to `docs/superpowers/specs/2026-10-22-regressor-centred-uq-diffusion-design.md`:

```markdown
## As built (plan `docs/superpowers/plans/2026-10-22-regressor-centred-uq-diffusion.md`)

- D1: the cached centre x̂_R travels in the existing `x0_hat` condition slot (`XRCachedData` returns `(x, y_d, x_r)` with `sigma_d = σ_R`); no separate `x_r` key. The checkpoint meta carries `sigma_r`, `centre`, `xr_cache_sha256`, `split_sha256`, `train_half`.
- D2: `varhead` takes `--loss_type nll` explicitly (`l1`/`l2` refused for it, `nll` refused for every other mode).
- D3: the cache's `val/` part holds all 1 000 test-split cubes centred by the full regressor; σ_full uses the 800 monitoring files; `uq_eval.py` recomputes x̂_R for the gate cubes in float32.
- D4: `uq_eval.py` runs without `--diffusion` / `--varhead` for the Stage 0 dry run.
- σ_R is the RMS (not the std) of the cross-fitted residual elements, the exact definition of σ_d.
```

- [ ] **Step 4: Run the whole test suite once**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests -q -x`
Expected: all pass (dataset-marked tests run on this machine).

- [ ] **Step 5: Commit**

```bash
git add scripts/pilot/smoke_uq.sh docs/superpowers/specs/2026-10-22-regressor-centred-uq-diffusion-design.md
git commit -F <message file>   # "test(uq): CPU end-to-end smoke of Phase 2a; spec as-built note (Phase 2a Task 10)" + trailers
```

---
## Execution tasks (GPU; each one is a pre-registered step of spec §7 — no gate number is looked at before its task says so)

Conventions for Tasks 11–17: `R=/data/chaoyi_he/HSI/Diffu/results/residual_warmstart`; `PY=/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python`; every launch goes through `scripts/pilot/launch_run.sh` (which refuses a busy GPU); long waits use a watcher script in the job's tmp directory (heartbeat ≤ 110 min, exits on DONE), re-armed until DONE; results and decisions are appended to `docs/superpowers/plans/2026-10-22-regressor-centred-uq-diffusion-RESULTS.md` (created in Task 11) and committed after each task.

### Task 11: Stage 0 dry run on the d = 4 assets (harness validation, no thresholds)

- [ ] **Step 1: Score the existing d = 4 assets** (GPU 0 when free; ≈ 20 min):

```bash
$PY scripts/uq_eval.py --d 4 --name stage0_d4_assets --centre $R/runs/regress_l1_sel_d4_b64/checkpoint_epoch_17.pth \
    --ensemble $R/runs/regress_l1_sel_d4_b64/checkpoint_epoch_17.pth $R/runs/regress_l1_sel_s43_d4_b64/checkpoint_epoch_17.pth $R/runs/regress_l2_sel_d4_b64/checkpoint_epoch_18.pth \
    --b $R/runs/residual_d4_b64/checkpoint_epoch_100.pth --K 16 --sampler ddim10 --device cuda:0 --batch_size 4
```

- [ ] **Step 2: Check the harness against known numbers.** In the record `means.regressor.rmse_pct` must be 1.9621 ± 0.001 and `means.baseline.rmse_pct` 2.3512 ± 0.001 (the gate records of the B-track); `means.b.single_rmse_pct` ≈ 2.2–2.4 (B's single DDIM-10 draw) and `means.b.mean_rmse_pct` below it (the mean of 16 draws sits near B's chain mean 2.11–2.2); `means.ensemble.cov90` is reported for the 3-member Gaussian. Any disagreement with the first two numbers is a harness bug: stop and fix the owning task.

- [ ] **Step 3: Create the results note** `docs/superpowers/plans/2026-10-22-regressor-centred-uq-diffusion-RESULTS.md` with a header (spec link, plan link), a "Stage 0" section holding the record path and the table of `means` per method (regressor, baseline, B single / mean-of-16 / CRPS / cov90 / AUSE, ensemble CRPS / cov90 / AUSE), and the sentence "No d = 8 number has been looked at." Commit (note only).

### Task 12: Stage 1 — split and the two half regressors at d = 8

- [ ] **Step 1: Write the split once** (CPU): `$PY scripts/uq/split_halves.py` → `$R/uq/split_halves.json`; record its sha256 (`sha256sum`) and the images-per-class line in the results note.

- [ ] **Step 2: Launch R₁ and R₂** (one per GPU, when both are free; ≈ 1.5 h each):

```bash
RUN_NAME=regress_l1_half1_d8_b64 bash scripts/pilot/launch_run.sh regress 0 --prediction_type direct --loss_type l1 --ds 8 --train_half 1 --num_epochs 40 --val_every 1 --save_every 1 --keep_last 40
RUN_NAME=regress_l1_half2_d8_b64 bash scripts/pilot/launch_run.sh regress 1 --prediction_type direct --loss_type l1 --ds 8 --train_half 2 --num_epochs 40 --val_every 1 --save_every 1 --keep_last 40
```

- [ ] **Step 3: Select** when both show DONE = 40: `$PY scripts/uq/select_epoch.py $R/runs/regress_l1_half1_d8_b64` and the same for half2; write both selections (epoch, val_loss, last_val_loss) into the results note. Then prune each half run to its selected epoch + 40 with the prune pattern of the B-track (dry-run first).

### Task 13: Stage 1 — the centre cache at d = 8

- [ ] **Step 1: Build** (GPU 0, ≈ 20 min; the full regressor is the pre-registered `regress_l1_sel_d8_b64@11`):

```bash
$PY scripts/uq/build_xr_cache.py --d 8 --half1 $R/runs/regress_l1_half1_d8_b64/checkpoint_epoch_<E1>.pth --half2 $R/runs/regress_l1_half2_d8_b64/checkpoint_epoch_<E2>.pth --full $R/runs/regress_l1_sel_d8_b64/checkpoint_epoch_11.pth --device cuda:0
```

- [ ] **Step 2: Record** σ_R, σ_full, the ratio and the flag from `$R/uq/d8/xr_cache/meta.json` in the results note (a flagged ratio is reported, not acted on); `du -sh` the cache (expect ≈ 5 GiB).

### Task 14: Stage 1 — the residual diffusion, the variance head and the second-seed regressor at d = 8

- [ ] **Step 1: Launch** (when the GPUs are free; the diffusion ≈ 8.5 h, the variance head ≈ 3 h):

```bash
RUN_NAME=residual_reg_d8_b64 bash scripts/pilot/launch_run.sh residual_reg 0 --ds 8 --xr_cache $R/uq/d8/xr_cache
RUN_NAME=varhead_d8_b64 bash scripts/pilot/launch_run.sh varhead 1 --prediction_type direct --loss_type nll --ds 8 --xr_cache $R/uq/d8/xr_cache --num_epochs 40 --val_every 1 --save_every 1 --keep_last 40
```

- [ ] **Step 2: When `varhead_d8_b64` is DONE** (GPU 1 free): select its epoch (`select_epoch.py`, lowest monitoring NLL), note it, prune to selected + 40, and launch the optional second seed for the ensemble baseline (≈ 7.5 h):

```bash
RUN_NAME=regress_l1_sel_s43_d8_b64 bash scripts/pilot/launch_run.sh regress 1 --prediction_type direct --loss_type l1 --ds 8 --seed 43 --val_every 1 --save_every 1 --keep_last 100
```

and when it is DONE select its epoch and prune to selected + 25/50/100.

- [ ] **Step 3: When `residual_reg_d8_b64` is DONE = 100**: the checkpoint evaluated is epoch 100 (diffusion runs are not selected; the pre-registration says 100 epochs). Note the final monitoring loss curve (val every 5) in the results note.

### Task 15: Stage 1 — evaluation and the one verdict at d = 8

- [ ] **Step 1: Evaluate** (GPU 0, ≈ 1 h for K = 32 with the half-centre sensitivity row and the ensemble):

```bash
$PY scripts/uq_eval.py --d 8 --name d8_main --centre $R/runs/regress_l1_sel_d8_b64/checkpoint_epoch_11.pth \
    --diffusion $R/runs/residual_reg_d8_b64/checkpoint_epoch_100.pth --varhead $R/runs/varhead_d8_b64/checkpoint_epoch_<EV>.pth \
    --ensemble $R/runs/regress_l1_sel_d8_b64/checkpoint_epoch_11.pth $R/runs/regress_l1_sel_s43_d8_b64/checkpoint_epoch_<ES>.pth \
    --centre_halves $R/runs/regress_l1_half1_d8_b64/checkpoint_epoch_<E1>.pth $R/runs/regress_l1_half2_d8_b64/checkpoint_epoch_<E2>.pth \
    --xr_cache $R/uq/d8/xr_cache --K 32 --sampler ddim10 --device cuda:0 --batch_size 4
```

(omit `--ensemble`'s second member if the seed-43 run was skipped; the ensemble is report-only.)

- [ ] **Step 2: The verdict, once:** `$PY scripts/uq_verdict.py $R/uq/d8/eval/d8_main.json`. Copy its printed table verbatim into the results note under "Stage 1 verdict (d = 8)", with the record and verdict paths and the `reported` block (C3, spread ratios, ensemble, half-centre row).

- [ ] **Step 3: The pre-registered fix round, only if `calibration_pass` is false:** one more evaluation with `--K 64 --name d8_fix_k64` and one with `--sampler ddpm1000 --subset ddpm --K 8 --name d8_fix_ddpm`, each followed by its verdict; the thresholds are unchanged and the results note states which (if any) passes. No other re-run.

- [ ] **Step 4: Commit** the results note and update memory (`residual-warmstart-plan.md` status and `phase1a-prime-followups.md`).

### Task 16: Stage 2 — the d = 4 contrast (report only)

- [ ] **Step 1:** Repeat Tasks 12–15 at `--ds 4` with the names `regress_l1_half1_d4_b64`, `regress_l1_half2_d4_b64`, `residual_reg_d4_b64`, `varhead_d4_b64`; the full centre is `regress_l1_sel_d4_b64@17`; the ensemble members are the three existing selected checkpoints; `--K 16`; `--b $R/runs/residual_d4_b64/checkpoint_epoch_100.pth` adds the linear-centred B row; the verdict runs with `--report_only`.

- [ ] **Step 2:** Results note: "Stage 2 (d = 4 contrast)" table next to the d = 8 one, and the dose–response sentence the spec asks for (is the CRPS gap and the texture gap smaller than at d = 8?).

### Task 17: Stage 3 — write-up

- [ ] **Step 1:** Finish the results note: conclusion against the spec's success criteria, caveats (σ ratio flag, S1 drift, K), what the user decides next (part (b)).

- [ ] **Step 2:** The GDN-style page (as `docs/analysis/phase1a_prime_page/` did it: extract → compact → build): perception–distortion table, calibration (coverage) and sparsification curves, the 8 stored cubes' sample / mean / std / truth figures at d = 8 and d = 4. Publish as an artifact, link it in the results note.

- [ ] **Step 3:** Plan as-built block appended to this plan; memory updated; commit; push. Then `superpowers:finishing-a-development-branch`.

---

## Self-review notes (written after the plan)

- **Spec coverage.** §4.1 → T1; §4.2 → T4 (`--train_half`), T6 (guard), T12; §4.3 → T3, T5, T13; §4.4 → T2, T4, T8 (`draw_samples`), T14; §4.5 → T2, T4, T14; §4.6 → T8 (`--ensemble`), T14 step 2; §4.7 → T8 (`--centre_halves`), T15; §5.1–5.6 → T7, T8; §5.7 → T9, T15; §6 → file structure table; §7 → T10 (smoke), T11–T17; §8 → each task's tests + T10; §9/§10 → T15 step 3 (fix round), T17 caveats. Gap found and closed: the spec's "Stage 0 reproduces 1.9621 % / 2.3512 %" check is T11 step 2.
- **Placeholders.** `<E1>`, `<E2>`, `<EV>`, `<ES>` in T13–T15 are the selected epochs that only exist after T12/T14 run; each is produced by `select_epoch.py` in the step before its use. No other TBDs.
- **Type consistency.** `checkpoint_extra()` keys (`sigma_r`, `xr_cache_sha256`, `centre`, `split_sha256`) are produced in T3, consumed in T4 (meta) and T8 (`check_models` reads `meta['centre']['sha256']`, `meta['xr_cache_sha256']`, `meta['sigma_r']`); `ckpt_info` (T5) writes `train_half` into the cache meta's `half1/half2/full`, which `check_crossfit_meta` (T3) reads; `PREREG_CENTRE` lives in T8 and is imported by T9; the record keys written by `run_eval` (T8: `per_cube.diffusion.{crps,cov50,cov90,single_hf,single_rmse_pct,mean_rmse_pct,ause,spread_ratio}`, `per_cube.varhead.{crps,ause,cov90,spread_ratio}`, `per_cube.regressor.{rmse_pct,hf}`, `per_cube.baseline.rmse_pct`, `yswap.{true,swap_full}`, `provenance.centre.{run,epoch}`) are exactly the ones `clauses` (T9) reads.
- **Review Focus.** Item 1 → `test_a_cache_of_another_stride_is_refused` (T3); item 2 → `test_ckpt_info_checks_role_mode_and_stride` (T5); item 3 → `test_resume_refuses_a_rebuilt_cache_and_another_half` (T4); item 4 → `test_crps_samples_refuses_fewer_than_two_samples` and the constant-sample coverage assertions (T7); item 5 → `test_centre_check_and_paths` and `test_main_writes_once_and_refuses_overwrite` (T9).
