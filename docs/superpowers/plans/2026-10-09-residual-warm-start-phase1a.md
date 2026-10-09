# Residual Warm-Start — Phase 1a (Pilot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the residual model B and its two ablations A and A′ at d = 4, `base_channels` 64, on this server, and decide with the Phase 0 harness whether B beats the linear baseline and the ablations at ≤ 10 NFE (spec §8 gate 1a).

**Architecture:** One new training entry point `main_2d_residual_fsdp.py` (single GPU or FSDP, bf16, exact resume, atomic checkpoints carrying the `meta` dict the harness verifies, every distributed decision made collectively) built on the Phase 0 pieces — `HFDResidualData`, `ResidualWrapper`, `DiffusionTrainer.get_loss` — and the FSDP conventions of `main_2d_fsdp.py` without its known bugs. The harness `scripts/eval_warmstart.py` gains the A′ mode, a quick sampler grid for milestone checks and collision-free output names. `scripts/gate_1a.py` pairs the three final full-grid evaluations cube by cube and refuses anything that is not a final full-grid evaluation. An idempotent driver `scripts/pilot/pilot_driver.py` runs the ~16 h pilot procedure (launch, resume after a crash, milestone and final evaluations, gate) and can be re-invoked at any time.

**Tech Stack:** Python 3.10 (`/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python`), torch 2.8.0+cu126, FSDP1 API, pytest; 2 × RTX A4500 (20 GB).

**Spec:** `docs/superpowers/specs/2026-10-08-residual-warm-start-design.md` (§4 method, §5 ablations, §7 evaluation protocol, §8 phases and gates, §9 safeguards, §11 risks). Phase 0 results and rulings: `docs/superpowers/plans/2026-10-08-residual-warm-start-phase0-RESULTS.md` (binding where it supersedes the spec).

## Deviations from the spec (for the user to approve with this plan)

1. **A′ ablation added** (Phase 0 review): `standard_xhat` = standard target x with x̂₀ concatenated to the input. The gate reports B − A′ next to the spec's B − baseline and B − A; verdict matrix in Task 4.
2. **Artefact layout**: runs under `results/residual_warmstart/runs/<mode>_d<d>_b<base>/`, eval JSON `results/residual_warmstart/eval/<mode>/d<d>/<run_name>__<ckpt>[__quick].json` (spec §12 says `results/residual_warmstart/<mode>/d<d>/…`; the Phase 0 ruling fixed `eval/<mode>/d<d>/<ckpt>.json`; this plan adds the run name and grid suffix so the pilot, the full run, A and A′ never collide).
3. **Milestone checkpoints** 25/50/100 are exempt from "keep the last two" (spec §6); checkpoints are also written every 10 epochs.
4. **No EMA** in any arm (spec silent; the existing pipeline has none; the user chose batch 4/GPU over a sharded EMA for the full model). *Open for the user: say so if you want an fp32 EMA in the pilot arms.*
5. **Quick-grid evaluations at epochs 25 and 50 share the GPU of their own run** (spec §9 "never start on a busy GPU" protects other jobs; a run's own GPU is ours); the final full-grid evaluation runs on an idle GPU.
6. **Pilot hyper-parameters** (spec fixes only "AdamW, cosine, clip 1.0, checkpoints every 10 epochs"): batch 8 on one GPU, lr 1e-4 → 0.033·lr, wd 0, eps 1e-6, v-prediction, L1, min-SNR 5, seed 42, identical for all three arms.

## Global Constraints

- `main_2d.py`, `main_2d_fsdp.py` and the sensor-SR pipeline are not modified (spec §10). `model/*` and `data_loader/*` change only additively; all 64 existing tests keep passing (one existing harness test is *edited* in Task 1 to follow the new output naming — listed there).
- Loader unchanged: `HFDResidualData(data_path, split, sensor_down_sample_rate=4, prior_path)` — 9000 training files, 1000 validation files, no sensor noise, d fixed for the whole run (no `--ds_warmup_epochs`).
- Dataset root `os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')`.
- **Artefact root** `RESULTS_ROOT = os.environ.get('RESIDUAL_WARMSTART_RESULTS', '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart')` — absolute, outside any worktree; the prior `RESULTS_ROOT/linear_prior_R1.npz` (sha256 `1ad9108f17bee02eceadf63e5cd59671fe3475a9b015aa8c7c9acfd5adaba2b0`, σ₄ = 0.051661), `RESULTS_ROOT/eval/`, `RESULTS_ROOT/runs/`, `RESULTS_ROOT/timing_phase1.json`. Code tasks (1, 2, 4, 5) run in the worktree; GPU tasks (3, 6) run from the main checkout `/data/chaoyi_he/HSI/Diffu` on `master` after the code tasks are merged. Deleting a worktree must never delete pilot output.
- Checkpoints hold the **`ResidualWrapper` state dict** (keys `net.*`) under `model_state_dict`, loaded `strict=True`, plus `meta = {mode, d, base_channels, prediction_type, n_timesteps=1000, sigma_d, prior_s, prior_sha256, args}`; `check_meta` compares `mode, d, base_channels, prediction_type` (exact), `sigma_d` (|Δ| < 1e-9), `prior_sha256`. T = 1000 is fixed (no `--timesteps` flag).
- Training target: residual `r = (x − x̂₀)/σ_d` for mode `residual`; `x` for `standard` and `standard_xhat`. Network input `[x_t, x̂₀]` for `residual` and `standard_xhat` (`extra_in_channels=64`, `use_x0_hat=True`), `x_t` alone for `standard`. Condition dict `{'sensor': y_d, 'x0_hat': x0_hat}` moved to the device **per key** (`get_loss` moves only `x_0`).
- Residual-chain sampling uses `clamp_x0 = (−2/σ_d, 2/σ_d)` (already in `reconstruct`); the final cube is clamped to [−1, 1].
- **Every distributed decision is collective**: checkpoint saves are called on every rank (only the write is rank-gated); the non-finite-loss skip/abort, the free-memory refusal and the wall-clock stop are all-reduced or broadcast before any rank acts on them.
- **Exact resume**: the randomness of epoch e depends only on (seed, rank, e) — per-epoch re-seeding and a `DistributedSampler` (also with one process) with `set_epoch` — so 2 + resume + 1 epochs equals 3 uninterrupted epochs bit for bit on CPU; optimizer and scheduler states are restored, never restarted.
- Budget: 100 epochs or 24 h wall-clock per run, whichever first; the final checkpoint for the gate is epoch 100 or the last completed epoch; the cosine schedule is not rescaled when the cap fires (recorded in the results note).
- No peeking (spec §7): the gate uses each run's final checkpoint (the epoch in its `DONE` file), full grid, seeds [0, 1]; epochs 25 and 50 are diagnostic (quick grid) and never select anything; no `best_model.pth`; the monitoring validation loss is computed on the 800 validation files **outside** the 200-cube gate subset.
- GPU use: `nvidia-smi` before every launch; never start a *training* run on a GPU with < 3 GB free or with another compute process; full-grid evaluations require an idle GPU.
- Run tests with `cd <repo> && CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests -q` (the GPUs may be training during Task 6); dataset tests carry `@pytest.mark.dataset`, prior tests skip without the npz. New unit tests run on CPU in seconds (tiny `base_channels`, synthetic tensors).
- Commit after every task with the attribution lines from the session reminder; never stage `__pycache__` files.

## Review Focus

1. **Collectives on every rank.** `main_2d_fsdp.py:620-642` deadlocks on 2 GPUs by calling the save under `if is_main_process()`. Test: `test_save_checkpoint_gathers_on_every_rank` (fake FSDP/dist recorders) and `test_nonfinite_skip_is_collective` (fake all-reduce forcing a skip) in Task 2; the 2-GPU FSDP smoke in Task 3.
2. **A′ evaluated as if it were A or B.** Test: Task 1 pins `build_residual_model` per mode and A′ weights refuse to load into A; `check_meta` rejects a mode mismatch (existing test); Task 4's gate refuses a JSON whose `provenance.mode` is wrong for its slot.
3. **Resume that changes the run.** Test: `test_resume_is_bitwise_exact` (3 epochs vs 2 + save/resume + 1) in Task 2.
4. **Gate fed the wrong evaluation.** A quick-grid or epoch-50 JSON, a b256 run or mismatched seeds would be gated silently. Test: Task 4's refusal tests (`grid`, `seeds`, `d`, `base_channels`, `prior_sha256`, `meta_checked`, DONE epoch).
5. **Loader-bound training.** 4 workers make the pilot ≈ 50 % slower than budgeted. Test: Task 3's 200-step dry run reports steady-state data wait; the pilot is launched only with a configuration whose data wait is < 10 % of the step time.

---

### Task 1: A′ mode, shared model builder, quick grid, collision-free eval names, artefact root

**Files:**
- Modify: `model/residual_wrapper.py` (append `MODES`, `build_residual_model`, `target_for`)
- Modify: `scripts/eval_warmstart.py` (`RESULTS_ROOT`, `build_model`, `sampler_grid`, `eval_output_path`, `checkpoint_eval`, CLI)
- Test: `tests/test_wrapper.py`, `tests/test_eval_harness.py` (3 new tests + edit `test_checkpoint_eval_records_per_cube_values`)

**Interfaces:**
- Consumes: `ResidualWrapper(net, use_x0_hat)`, `U2NetHyperspectral(spectral_channels, sensor_channels, base_channels, extra_in_channels)`.
- Produces: `MODES = ('residual', 'standard', 'standard_xhat')`; `build_residual_model(mode, base_channels, spectral_channels=64, sensor_channels=30) -> ResidualWrapper` (CPU, train mode); `target_for(mode, x, x0_hat, sigma_d)`; `RESULTS_ROOT`, `PRIOR = RESULTS_ROOT/linear_prior_R1.npz`, `EVAL_DIR = RESULTS_ROOT/eval`; `sampler_grid(which='full')` with `'quick'` = `ddim1, ddim2, ddim5, ddim10, ddim20, warm{50,100,200,400}_ddim`; `eval_output_path(mode, d, run_name, ckpt, grid='full') -> str`; CLI `--mode {residual,standard,standard_xhat}`, `--grid {full,quick}`, `--run_name`; JSON gains top-level `grid` and `run_name`.

- [ ] **Step 1: Write the failing tests and edit the existing one**

Append to `tests/test_wrapper.py`:

```python
import pytest
import torch

from model.residual_wrapper import MODES, ResidualWrapper, build_residual_model, target_for


@pytest.mark.parametrize('mode,in_ch,use_xhat', [('residual', 128, True), ('standard', 64, False), ('standard_xhat', 128, True)])
def test_build_residual_model_per_mode(mode, in_ch, use_xhat):
    m = build_residual_model(mode, base_channels=8)
    assert isinstance(m, ResidualWrapper) and m.use_x0_hat is use_xhat
    assert m.net.input_proj.in_channels == in_ch
    assert all(k.startswith('net.') for k in m.state_dict())
    x_t = torch.randn(1, 64, 16, 16)
    cond = {'sensor': torch.randn(1, 30, 4, 4), 'x0_hat': torch.randn(1, 64, 16, 16)}
    assert m(x_t, cond, torch.tensor([3])).shape == x_t.shape


def test_build_residual_model_rejects_unknown_mode():
    with pytest.raises(ValueError):
        build_residual_model('standard_x0hat', base_channels=8)
    assert MODES == ('residual', 'standard', 'standard_xhat')


def test_target_for_modes():
    x = torch.rand(2, 4, 8, 8) * 2 - 1
    xh = torch.rand(2, 4, 8, 8) * 2 - 1
    assert torch.allclose(target_for('residual', x, xh, 0.05), (x - xh) / 0.05)
    assert torch.equal(target_for('standard', x, xh, 0.05), x)
    assert torch.equal(target_for('standard_xhat', x, xh, 0.05), x)
```

Append to `tests/test_eval_harness.py`:

```python
from scripts.eval_warmstart import build_model, eval_output_path, sampler_grid


def test_sampler_grid_quick_is_the_low_nfe_subset():
    full = {c['name']: c for c in sampler_grid('full')}
    quick = sampler_grid('quick')
    names = [c['name'] for c in quick]
    assert names == ['ddim1', 'ddim2', 'ddim5', 'ddim10', 'ddim20', 'warm50_ddim', 'warm100_ddim', 'warm200_ddim', 'warm400_ddim']
    assert all(c == full[c['name']] for c in quick)
    assert sampler_grid() == sampler_grid('full')
    with pytest.raises(ValueError):
        sampler_grid('fast')


def test_build_model_standard_xhat_matches_wrapper_builder():
    m = build_model('standard_xhat', 8, torch.device('cpu'))
    assert m.use_x0_hat and m.net.input_proj.in_channels == 128 and not m.training
    sd = m.state_dict()
    build_model('standard_xhat', 8, torch.device('cpu')).load_state_dict(sd, strict=True)
    with pytest.raises(RuntimeError):                      # A′ weights do not fit A
        build_model('standard', 8, torch.device('cpu')).load_state_dict(sd, strict=True)


def test_eval_output_path_separates_runs_and_grids():
    a = eval_output_path('standard', 4, 'standard_d4_b64', '/x/runs/standard_d4_b64/checkpoint_epoch_100.pth')
    b = eval_output_path('standard', 4, 'standard_d4_b256', '/x/runs/standard_d4_b256/checkpoint_epoch_100.pth')
    q = eval_output_path('standard', 4, 'standard_d4_b64', '/x/runs/standard_d4_b64/checkpoint_epoch_100.pth', grid='quick')
    assert a != b and a.endswith('eval/standard/d4/standard_d4_b64__checkpoint_epoch_100.json')
    assert q.endswith('eval/standard/d4/standard_d4_b64__checkpoint_epoch_100__quick.json')
    assert eval_output_path('standard_xhat', 4, None, '/x/runs/standard_xhat_d4_b64/checkpoint_epoch_100.pth') \
        .endswith('eval/standard_xhat/d4/standard_xhat_d4_b64__checkpoint_epoch_100.json')
```

Edit `test_checkpoint_eval_records_per_cube_values` (it exercises the real `checkpoint_eval`, so it must follow the new wiring): the `sampler_grid` stub becomes `lambda which='full': [...]`; the `Namespace` gains `grid='full', run_name='tiny_run'`; the output is read from `tmp_path / 'residual' / 'd4' / 'tiny_run__tiny.json'`; add `assert out['grid'] == 'full' and out['run_name'] == 'tiny_run'`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_wrapper.py tests/test_eval_harness.py -q`
Expected: collection errors (`ImportError`: `MODES`, `build_residual_model`, `eval_output_path`).

- [ ] **Step 3: Implement**

Append to `model/residual_wrapper.py`:

```python
MODES = ('residual', 'standard', 'standard_xhat')
# residual      (B):  input [x_t, x0_hat], target (x - x0_hat) / sigma_d
# standard      (A):  input  x_t,          target  x
# standard_xhat (A'): input [x_t, x0_hat], target  x   -- B's inputs with A's target: controls for the spatially
#                     aligned x0_hat channels that the position-blind cross-attention context cannot provide


def build_residual_model(mode, base_channels, spectral_channels=64, sensor_channels=30):
    """The one place that maps a mode name to an architecture; training and evaluation both call it."""
    from model.u2net_hyperspectral import U2NetHyperspectral      # local import: u2net_hyperspectral must not import this module
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
    use_x0_hat = mode in ('residual', 'standard_xhat')
    net = U2NetHyperspectral(spectral_channels=spectral_channels, sensor_channels=sensor_channels,
                             base_channels=base_channels, extra_in_channels=spectral_channels if use_x0_hat else 0)
    return ResidualWrapper(net, use_x0_hat=use_x0_hat)


def target_for(mode, x, x0_hat, sigma_d):
    """The diffusion target per mode (all on the loader's [-1, 1] scale)."""
    if mode == 'residual':
        return (x - x0_hat) / sigma_d
    return x
```

In `scripts/eval_warmstart.py`:

```python
from model.residual_wrapper import MODES, build_residual_model  # noqa: E402  (replaces the ResidualWrapper import; the U2NetHyperspectral import becomes unused and is removed)

RESULTS_ROOT = os.environ.get('RESIDUAL_WARMSTART_RESULTS', '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart')
PRIOR = os.path.join(RESULTS_ROOT, 'linear_prior_R1.npz')          # replaces the REPO-relative PRIOR
EVAL_DIR = os.path.join(RESULTS_ROOT, 'eval')                       # replaces the REPO-relative EVAL_DIR

def build_model(mode, base_channels, device):
    return build_residual_model(mode, base_channels).to(device).eval()

QUICK_GRID = ('ddim1', 'ddim2', 'ddim5', 'ddim10', 'ddim20', 'warm50_ddim', 'warm100_ddim', 'warm200_ddim', 'warm400_ddim')

def sampler_grid(which='full'):
    g = [dict(name='ddpm1000', method='ddpm', n_steps=1000)]
    g += [dict(name=f'ddim{k}', method='ddim', n_steps=k) for k in (1, 2, 5, 10, 20)]
    g += [dict(name=f'warm{t}_ddim', method='ddim', n_steps=10, t_start=t) for t in (50, 100, 200, 400)]
    g += [dict(name=f'warm{t}_ddpm', method='ddpm', n_steps=1000, t_start=t) for t in (50, 100, 200, 400)]
    if which == 'full':
        return g
    if which == 'quick':                         # milestone checks: every row costs <= 20 NFE, no 50-cube DDPM row
        return [c for c in g if c['name'] in QUICK_GRID]
    raise ValueError(f"grid must be 'full' or 'quick', got {which!r}")

def eval_output_path(mode, d, run_name, ckpt, grid='full'):
    """eval/<mode>/d<d>/<run_name>__<ckpt stem>[__<grid>].json. run_name defaults to the checkpoint's parent directory
    name, so the pilot (…_b64) and the full run (…_b256) of one mode and d never overwrite each other; a non-full grid
    is suffixed so a quick milestone evaluation never overwrites the final full one."""
    run_name = run_name or os.path.basename(os.path.dirname(os.path.abspath(ckpt)))
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    suffix = '' if grid == 'full' else f'__{grid}'
    return os.path.join(EVAL_DIR, mode, f'd{d}', f'{run_name}__{stem}{suffix}.json')
```

In `checkpoint_eval`: `grid = getattr(args, 'grid', 'full'); run_name = getattr(args, 'run_name', None) or os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))`; `for cfg in sampler_grid(grid):`; add `grid=grid, run_name=run_name` to `out`; replace the `out_dir`/`path` lines with `path = eval_output_path(args.mode, args.d, run_name, args.ckpt, grid); os.makedirs(os.path.dirname(path), exist_ok=True)`. In `main`: `ap.add_argument('--mode', choices=list(MODES))`, `ap.add_argument('--grid', choices=['full', 'quick'], default='full')`, `ap.add_argument('--run_name', default=None)`. Update the module docstring: the usage line mentions `--grid quick` and the output-path sentence becomes `eval/<mode>/d<d>/<run_name>__<ckpt>[__quick].json`. `reconstruct` and `yswap_check` need no change: every non-`residual` mode is "target x, `x_init = x0_hat`, default clamp", which is exactly A′. The Phase 0 baseline files stay at `EVAL_DIR/baseline_d<d>.json` — now resolved under `RESULTS_ROOT`, where the Phase 0 artefacts already live in the main checkout.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests -q`
Expected: 72 passed (64 existing + 5 wrapper items + 3 harness tests), no skips on this server. (The 13 prior-dependent tests read the repo-relative `results/residual_warmstart/linear_prior_R1.npz`; in a fresh worktree copy it from the main checkout first, otherwise they skip.)

- [ ] **Step 5: Commit**

```bash
git add model/residual_wrapper.py scripts/eval_warmstart.py tests/test_wrapper.py tests/test_eval_harness.py
git commit -m "feat(warmstart): standard_xhat ablation mode, shared model builder, quick sampler grid, run-named eval outputs under an absolute results root

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
```

---

### Task 2: Training entry point `main_2d_residual_fsdp.py`

**Files:**
- Create: `main_2d_residual_fsdp.py`
- Test: `tests/test_train_residual.py`

**Interfaces:**
- Consumes: `HFDResidualData`, `residual_collate_fn`, `build_residual_model`, `target_for`, `DiffusionTrainer(device, loss_type, prediction_type, snr_gamma).get_loss(model, x_0, cond, loss_in_fp32=True) -> (loss, info)`, `scripts.eval_warmstart.check_meta` (test only), `tests/data/val_subset_200.json`.
- Produces: a script runnable as `python main_2d_residual_fsdp.py --mode residual --device cuda:0 --save_dir <dir>` (single GPU) or `torchrun --standalone --nproc_per_node=2 main_2d_residual_fsdp.py --mode residual --save_dir <dir>` (FSDP); a run directory with `args.json`, `train_log.jsonl` (one JSON line per epoch), `checkpoint_epoch_<N>.pth` (atomic; last two + `--keep_epochs`), `DONE` (final epoch number); importable functions used by the tests, Task 3 and Task 5: `get_parser()`, `NO_DIST`, `setup_dist(args)`, `seed_everything(seed, rank)`, `epoch_seed(args, epoch, rank)`, `make_datasets(args)`, `make_loader(ds, args, batch_size, shuffle, rank, world, device)`, `make_trainer(args, device)`, `make_optimizer(model, args)`, `make_scheduler(optimizer, args)`, `checkpoint_meta(args, sigma_d, prior_s, prior_sha256)`, `save_checkpoint(model, optimizer, scheduler, epoch, train_loss, meta, path, elapsed_hours, dist_ctx)`, `load_checkpoint(path)`, `restore_training_state(ck, model, optimizer, scheduler, dist_ctx)`, `latest_checkpoint(run_dir)`, `prune_checkpoints(run_dir, keep_last, keep_epochs)`, `should_save(epoch1, args)`, `all_min(value, device, dist_ctx)`, `train_one_epoch(model, trainer, loader, optimizer, args, device, dist_ctx, mode, sigma_d, log_prefix='')`, `validate(model, trainer, loader, args, device, dist_ctx, mode, sigma_d)`, `run_epochs(args, parts, dist_ctx, device)`, `dry_run(args)`, `main(args)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_train_residual.py`:

```python
"""CPU tests of the training entry point on a tiny synthetic dataset (no HFD files, no GPU)."""
import contextlib
import os
from types import SimpleNamespace

import pytest
import torch

import main_2d_residual_fsdp as tr
from model.residual_wrapper import build_residual_model

CPU = torch.device('cpu')


class TinyResidualSet(torch.utils.data.Dataset):
    """Stands in for HFDResidualData: (x64, y_d, x0_hat) as HWC float32 numpy arrays on the [-1, 1] scale."""
    sigma_d = 0.05
    prior = {'s': 3e-7}

    def __init__(self, n=8, H=16, d=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, H, H, 64, generator=g) * 2 - 1
        self.y = torch.rand(n, H // d, H // d, 30, generator=g)
        self.xh = (self.x + 0.05 * torch.randn(n, H, H, 64, generator=g)).clamp(-1, 1)
        self.img_list = [f'MatFlower60/Train/P0{22 + i % 3}/f{i}.mat' for i in range(n)]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        return self.x[i].numpy(), self.y[i].numpy(), self.xh[i].numpy()


def tiny_args(tmp_path, **kw):
    a = tr.get_parser().parse_args(['--mode', 'residual', '--device', 'cpu', '--base_channels', '8', '--batch_size', '2',
                                    '--num_workers', '0', '--num_epochs', '3', '--save_dir', str(tmp_path / 'run'), '--no_bf16'])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def make_parts(args, mode='residual', ds=None):
    ds = ds or TinyResidualSet()
    loader = tr.make_loader(ds, args, batch_size=args.batch_size, shuffle=True, rank=0, world=1, device=CPU)
    model = build_residual_model(mode, args.base_channels)
    trainer = tr.make_trainer(args, CPU)
    opt = tr.make_optimizer(model, args)
    sched = tr.make_scheduler(opt, args)
    return ds, loader, model, trainer, opt, sched


def one_epoch(parts, args, epoch=0, mode='residual'):
    ds, loader, model, trainer, opt, sched = parts
    tr.epoch_seed(args, epoch, 0); loader.sampler.set_epoch(epoch)
    rec = tr.train_one_epoch(model, trainer, loader, opt, args, CPU, tr.NO_DIST, mode, ds.sigma_d)
    sched.step()
    return rec


def test_train_one_epoch_runs_and_reports(tmp_path):
    args = tiny_args(tmp_path)
    rec = one_epoch(make_parts(args), args)
    assert set(rec) >= {'loss', 'lr', 'steps', 'skipped_nonfinite', 'sec', 'data_sec'}
    assert rec['steps'] == 4 and rec['skipped_nonfinite'] == 0 and torch.isfinite(torch.tensor(rec['loss']))


def test_cond_dict_keys_reach_get_loss(tmp_path, monkeypatch):
    args = tiny_args(tmp_path)
    parts = make_parts(args); trainer = parts[3]
    seen = {}
    real = trainer.get_loss

    def spy(m, x0, cond, **kw):
        seen['keys'] = sorted(cond); seen['kw'] = kw
        return real(m, x0, cond, **kw)
    monkeypatch.setattr(trainer, 'get_loss', spy)
    one_epoch(parts, args)
    assert seen['keys'] == ['sensor', 'x0_hat'] and seen['kw'] == {'loss_in_fp32': True}


@pytest.mark.parametrize('mode', ['residual', 'standard', 'standard_xhat'])
def test_target_per_mode_reaches_get_loss(tmp_path, monkeypatch, mode):
    args = tiny_args(tmp_path, mode=mode)
    parts = make_parts(args, mode); trainer = parts[3]
    got = {}
    real = trainer.get_loss

    def spy(m, x0, cond, **kw):
        got.setdefault('x0', x0.clone())
        return real(m, x0, cond, **kw)
    monkeypatch.setattr(trainer, 'get_loss', spy)
    one_epoch(parts, args, mode=mode)
    assert (got['x0'].abs().max() > 1.5) if mode == 'residual' else (got['x0'].abs().max() <= 1.0)


def test_save_load_roundtrip_meta_and_harness_check(tmp_path):
    from scripts.eval_warmstart import check_meta
    args = tiny_args(tmp_path)
    parts = make_parts(args); ds, _, model, _, opt, sched = parts
    one_epoch(parts, args)
    meta = tr.checkpoint_meta(args, ds.sigma_d, ds.prior['s'], 'deadbeef' * 8)
    path = tmp_path / 'run' / 'checkpoint_epoch_1.pth'
    tr.save_checkpoint(model, opt, sched, 0, 0.5, meta, str(path), 0.01, tr.NO_DIST)
    assert path.exists() and not [f for f in os.listdir(path.parent) if f.endswith('.tmp')]
    ck = tr.load_checkpoint(str(path))
    assert set(ck) >= {'epoch', 'loss', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'meta', 'elapsed_hours'}
    assert all(k.startswith('net.') for k in ck['model_state_dict'])
    assert ck['meta'] == dict(mode='residual', d=4, base_channels=8, prediction_type='v', n_timesteps=1000,
                              sigma_d=ds.sigma_d, prior_s=3e-7, prior_sha256='deadbeef' * 8, args=vars(args))
    fresh = build_residual_model('residual', 8)
    fresh.load_state_dict(ck['model_state_dict'], strict=True)
    assert all(torch.equal(a, b) for a, b in zip(fresh.parameters(), model.parameters()))
    eval_args = SimpleNamespace(ckpt=str(path), mode='residual', d=4, base_channels=8, prediction_type='v')
    assert check_meta(ck, eval_args, ds.sigma_d, 'deadbeef' * 8) is True            # the harness accepts what training wrote
    with pytest.raises(ValueError, match='base_channels'):
        check_meta(ck, SimpleNamespace(**dict(vars(eval_args), base_channels=16)), ds.sigma_d, 'deadbeef' * 8)


def test_save_checkpoint_is_atomic(tmp_path, monkeypatch):
    args = tiny_args(tmp_path)
    ds, _, model, _, opt, sched = make_parts(args)
    meta = tr.checkpoint_meta(args, ds.sigma_d, 3e-7, 'x' * 64)
    path = str(tmp_path / 'run' / 'checkpoint_epoch_1.pth')
    tr.save_checkpoint(model, opt, sched, 0, 0.5, meta, path, 0.0, tr.NO_DIST)
    good = open(path, 'rb').read()

    def broken_save(obj, f, **kw):
        open(f, 'wb').write(b'partial'); raise OSError('disk full')
    monkeypatch.setattr(tr.torch, 'save', broken_save)
    with pytest.raises(OSError):
        tr.save_checkpoint(model, opt, sched, 1, 0.4, meta, path, 0.1, tr.NO_DIST)
    assert open(path, 'rb').read() == good                                          # the old checkpoint survives


def test_resume_is_bitwise_exact(tmp_path):
    """2 epochs + save + fresh process + restore + 1 epoch == 3 uninterrupted epochs (Review Focus 3)."""
    args = tiny_args(tmp_path, num_epochs=3)
    torch.manual_seed(0)
    parts = make_parts(args); ds, _, model, _, opt, sched = parts
    for e in range(2):
        one_epoch(parts, args, e)
    path = str(tmp_path / 'run' / 'checkpoint_epoch_2.pth')
    tr.save_checkpoint(model, opt, sched, 1, 0.4, tr.checkpoint_meta(args, ds.sigma_d, 3e-7, 'x' * 64), path, 0.2, tr.NO_DIST)
    one_epoch(parts, args, 2)                                                        # the uninterrupted third epoch
    ref = [p.detach().clone() for p in model.parameters()]
    ck = tr.load_checkpoint(path)
    parts2 = make_parts(args); ds2, _, model2, _, opt2, sched2 = parts2
    model2.load_state_dict(ck['model_state_dict'], strict=True)
    start_epoch, elapsed = tr.restore_training_state(ck, model2, opt2, sched2, tr.NO_DIST)
    assert start_epoch == 2 and elapsed == 0.2 and sched2.last_epoch == 2
    one_epoch(parts2, args, start_epoch)
    assert all(torch.equal(a, b) for a, b in zip(ref, model2.parameters()))


def test_prune_keeps_last_two_and_milestones(tmp_path):
    run = tmp_path / 'run'; run.mkdir()
    for e in (10, 20, 25, 30, 40, 50, 60):
        (run / f'checkpoint_epoch_{e}.pth').write_bytes(b'x')
    (run / 'args.json').write_text('{}')
    tr.prune_checkpoints(str(run), keep_last=2, keep_epochs=(25, 50, 100))
    assert sorted(f for f in os.listdir(run) if f.endswith('.pth')) == ['checkpoint_epoch_25.pth', 'checkpoint_epoch_50.pth', 'checkpoint_epoch_60.pth']


def test_nonfinite_loss_is_skipped_without_a_step_then_aborts(tmp_path, monkeypatch):
    args = tiny_args(tmp_path, max_nonfinite=10)
    parts = make_parts(args); ds, loader, model, trainer, opt, sched = parts
    before = [p.detach().clone() for p in model.parameters()]
    real = trainer.get_loss; calls = {'n': 0}

    def nan_first(m, x0, cond, **kw):
        calls['n'] += 1
        return (torch.tensor(float('nan'), requires_grad=True), {}) if calls['n'] == 1 else real(m, x0, cond, **kw)
    monkeypatch.setattr(trainer, 'get_loss', nan_first)
    rec = one_epoch(parts, args)
    assert rec['skipped_nonfinite'] == 1 and rec['steps'] == 3
    monkeypatch.setattr(trainer, 'get_loss', lambda m, x0, cond, **kw: (torch.tensor(float('nan'), requires_grad=True), {}))
    args.max_nonfinite = 2
    with pytest.raises(RuntimeError, match='non-finite'):
        one_epoch(parts, args, 1)
    # an all-NaN epoch shorter than max_nonfinite must not touch the parameters
    parts3 = make_parts(args); model3 = parts3[2]; before3 = [p.detach().clone() for p in model3.parameters()]
    args.max_nonfinite = 100
    monkeypatch.setattr(parts3[3], 'get_loss', lambda m, x0, cond, **kw: (torch.tensor(float('nan'), requires_grad=True), {}))
    assert one_epoch(parts3, args)['steps'] == 0
    assert all(torch.equal(a, b) for a, b in zip(before3, model3.parameters()))


def test_nonfinite_skip_is_collective(tmp_path, monkeypatch):
    """A finite loss on this rank is still skipped when the all-reduced flag says another rank was non-finite."""
    args = tiny_args(tmp_path)
    parts = make_parts(args); ds, loader, model, trainer, opt, sched = parts
    before = [p.detach().clone() for p in model.parameters()]
    fake_ctx = SimpleNamespace(enabled=True, rank=0, world=2, local_rank=0)

    def all_reduce(t, op=None):
        t.zero_()                                            # "some rank saw NaN"
    monkeypatch.setattr(tr.dist, 'all_reduce', all_reduce)
    monkeypatch.setattr(tr.dist, 'ReduceOp', SimpleNamespace(MIN=0, SUM=1))
    tr.epoch_seed(args, 0, 0); loader.sampler.set_epoch(0)
    rec = tr.train_one_epoch(model, trainer, loader, opt, args, CPU, fake_ctx, 'residual', ds.sigma_d)
    assert rec['steps'] == 0 and rec['skipped_nonfinite'] == 4
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_save_checkpoint_gathers_on_every_rank(tmp_path, monkeypatch):
    """Under FSDP the state-dict gathers and the barrier must run on every rank; only the write is rank-gated."""
    args = tiny_args(tmp_path)
    ds, _, model, _, opt, sched = make_parts(args)
    events = []

    class FakeFSDP:
        @staticmethod
        @contextlib.contextmanager
        def state_dict_type(m, t, *cfgs):
            events.append('gather'); yield

        @staticmethod
        def optim_state_dict(m, o):
            events.append('optim'); return o.state_dict()
    monkeypatch.setattr(tr, 'FSDP', FakeFSDP)
    monkeypatch.setattr(tr.dist, 'barrier', lambda: events.append('barrier'))
    meta = tr.checkpoint_meta(args, ds.sigma_d, 3e-7, 'x' * 64)
    for rank in (1, 0):
        ctx = SimpleNamespace(enabled=True, rank=rank, world=2, local_rank=rank)
        tr.save_checkpoint(model, opt, sched, 0, 0.5, meta, str(tmp_path / f'r{rank}.pth'), 0.0, ctx)
    assert events == ['gather', 'optim', 'barrier', 'gather', 'optim', 'barrier']
    assert not (tmp_path / 'r1.pth').exists() and (tmp_path / 'r0.pth').exists()


def test_run_epochs_calls_save_on_every_rank(tmp_path, monkeypatch):
    """The call site in the epoch loop is not rank-gated (fake rank 1 still reaches save_checkpoint)."""
    args = tiny_args(tmp_path, num_epochs=1, save_every=1, keep_epochs=[], val_every=0)
    parts = make_parts(args)
    calls = []
    monkeypatch.setattr(tr, 'save_checkpoint', lambda *a, **k: calls.append(a[6]))
    fake_ctx = SimpleNamespace(enabled=False, rank=1, world=2, local_rank=1)         # enabled=False: no collectives on CPU
    tr.run_epochs(args, parts, fake_ctx, CPU)
    assert calls and calls[0].endswith('checkpoint_epoch_1.pth')
    assert not os.path.exists(tmp_path / 'run' / 'train_log.jsonl')                 # rank 1 writes no logs


def test_validate_is_deterministic_and_reduced(tmp_path, monkeypatch):
    args = tiny_args(tmp_path)
    ds, _, model, trainer, _, _ = make_parts(args)
    vloader = tr.make_loader(ds, args, batch_size=2, shuffle=False, rank=0, world=1, device=CPU)
    a = tr.validate(model, trainer, vloader, args, CPU, tr.NO_DIST, 'residual', ds.sigma_d)
    torch.manual_seed(999); torch.rand(3)
    b = tr.validate(model, trainer, vloader, args, CPU, tr.NO_DIST, 'residual', ds.sigma_d)
    assert a['val_loss'] == pytest.approx(b['val_loss']) and a['val_n'] == 8
    monkeypatch.setattr(tr.dist, 'all_reduce', lambda t, op=None: t.mul_(2))           # "the other rank saw the same"
    monkeypatch.setattr(tr.dist, 'ReduceOp', SimpleNamespace(MIN=0, SUM=1))
    c = tr.validate(model, trainer, vloader, args, CPU, SimpleNamespace(enabled=True, rank=0, world=2, local_rank=0), 'residual', ds.sigma_d)
    assert c['val_n'] == 16 and c['val_loss'] == pytest.approx(a['val_loss'])


def test_milestones_schedule(tmp_path):
    args = tiny_args(tmp_path, num_epochs=100, save_every=10, keep_epochs=[25, 50, 100])
    assert [e for e in range(1, 101) if tr.should_save(e, args)] == sorted(set(list(range(10, 101, 10)) + [25, 50, 100]))


def test_make_datasets_excludes_gate_cubes_from_validation(tmp_path, monkeypatch):
    """The monitoring loss never sees the 200 gate cubes (no peeking)."""
    import json
    sub = json.load(open(os.path.join(tr.REPO, 'tests', 'data', 'val_subset_200.json')))
    fake_train = TinyResidualSet(n=4); fake_val = TinyResidualSet(n=8)
    fake_val.img_list = [os.path.join('/root', f) for f in sub['files'][:5]] + ['/root/MatFlower60/Train/P023/other.mat'] * 3
    monkeypatch.setattr(tr, 'HFDResidualData', lambda root, split, **kw: fake_train if split == 'train' else fake_val)
    args = tiny_args(tmp_path, data_root='/root')
    train, val = tr.make_datasets(args)
    assert len(train) == 4 and len(val) == 3 and all('other' in f for f in val.img_list)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_train_residual.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'main_2d_residual_fsdp'`.

- [ ] **Step 3: Write the training script**

`main_2d_residual_fsdp.py`:

```python
"""Residual warm-start diffusion training (spec 2026-10-08, Phase 1).

Single GPU:  python main_2d_residual_fsdp.py --mode residual --ds 4 --base_channels 64 --batch_size 8 --device cuda:0 \
                 --save_dir /data/chaoyi_he/HSI/Diffu/results/residual_warmstart/runs/residual_d4_b64
FSDP:        torchrun --standalone --nproc_per_node=2 main_2d_residual_fsdp.py --mode residual --ds 4 --base_channels 256 \
                 --batch_size 4 --save_dir .../runs/residual_d4_b256
Timing only: add --dry_run_steps 200 (sec/step with and without data wait, peak memory; writes nothing).

Modes (model/residual_wrapper.py): residual (B), standard (A), standard_xhat (A'). --batch_size is PER GPU.
Conventions taken from main_2d_fsdp.py (FSDP wrap, bf16, AdamW + cosine per epoch, FULL_STATE_DICT checkpoints) without
its rank-0-only checkpoint calls, worker cap and save-path rewriting. Every distributed decision (non-finite skip, memory
refusal, wall-clock stop) is collective. No ds warm-up: d is fixed because x0_hat and sigma_d depend on it.
"""
import argparse
import functools
import hashlib
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from data_loader.hfd_residual import HFDResidualData, residual_collate_fn  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from model.residual_wrapper import MODES, build_residual_model, target_for  # noqa: E402

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')
RESULTS_ROOT = os.environ.get('RESIDUAL_WARMSTART_RESULTS', '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart')
PRIOR = os.path.join(RESULTS_ROOT, 'linear_prior_R1.npz')
SUBSET = os.path.join(REPO, 'tests', 'data', 'val_subset_200.json')
N_TIMESTEPS = 1000
NO_DIST = SimpleNamespace(enabled=False, rank=0, world=1, local_rank=0)


def get_parser():
    ap = argparse.ArgumentParser(description="Residual warm-start diffusion training (B / A / A')")
    ap.add_argument('--mode', choices=list(MODES), required=True)
    ap.add_argument('--ds', '--sensor_down_sample_rate', dest='d', type=int, default=4, help='sensor downsampling d (fixed for the run)')
    ap.add_argument('--data_root', default=DATA_ROOT); ap.add_argument('--prior', default=PRIOR)
    ap.add_argument('--limit_files', type=int, default=0, help='debug: use only the first N training files (0 = all)')
    ap.add_argument('--base_channels', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=8, help='batch size PER GPU')
    ap.add_argument('--num_workers', type=int, default=8, help='DataLoader workers per process (>= 8 keeps the pilot compute-bound)')
    ap.add_argument('--num_epochs', type=int, default=100)
    ap.add_argument('--max_hours', type=float, default=24.0, help='stop after the epoch that crosses this wall-clock budget')
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--lrf', type=float, default=0.033, help='eta_min = lr * lrf')
    ap.add_argument('--weight_decay', type=float, default=0.0); ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--prediction_type', choices=['eps', 'x0', 'v'], default='v')
    ap.add_argument('--loss_type', choices=['l1', 'l2'], default='l1'); ap.add_argument('--snr_gamma', type=float, default=5.0)
    ap.add_argument('--val_every', type=int, default=5, help='loss-only validation on the non-gate validation files every N epochs (0 = never)')
    ap.add_argument('--save_every', type=int, default=10); ap.add_argument('--keep_last', type=int, default=2)
    ap.add_argument('--keep_epochs', type=int, nargs='*', default=[25, 50, 100], help='milestone checkpoints exempt from pruning')
    ap.add_argument('--max_nonfinite', type=int, default=10, help='abort after this many consecutive non-finite losses')
    ap.add_argument('--log_interval', type=int, default=200)
    ap.add_argument('--save_dir', required=True); ap.add_argument('--resume', default='', help='checkpoint path; "auto" = latest in --save_dir')
    ap.add_argument('--device', default='cuda:0', help='single-process device; ignored under torchrun')
    ap.add_argument('--no_bf16', action='store_true'); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sharding_strategy', choices=['FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD'], default='FULL_SHARD')
    ap.add_argument('--wrap_min_params', type=int, default=1_000_000)
    ap.add_argument('--dry_run_steps', type=int, default=0, help='time N steps (after warm-up) and exit without saving')
    return ap


# ----------------------------------------------------------------------------- setup
def setup_dist(args):
    """torchrun sets WORLD_SIZE; without it this is a single-process run on args.device."""
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        dist.init_process_group('nccl')
        local = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local)
        return SimpleNamespace(enabled=True, rank=dist.get_rank(), world=dist.get_world_size(), local_rank=local), torch.device(f'cuda:{local}')
    dev = torch.device(args.device)
    if dev.type == 'cuda':
        torch.cuda.set_device(dev)
    return NO_DIST, dev


def seed_everything(seed, rank):
    torch.manual_seed(seed + rank); np.random.seed(seed + rank); random.seed(seed + rank); torch.cuda.manual_seed_all(seed + rank)


def epoch_seed(args, epoch, rank):
    """The noise stream of epoch e depends only on (seed, rank, e): a resumed epoch replays nothing."""
    seed_everything(args.seed + 1000 * (epoch + 1), rank)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def all_min(value, device, dist_ctx):
    """Collective minimum of a scalar (identity without distributed)."""
    t = torch.tensor(float(value), device=device)
    if dist_ctx.enabled:
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return t.item()


def make_datasets(args):
    train = HFDResidualData(args.data_root, split='train', sensor_down_sample_rate=args.d, prior_path=args.prior)
    val = HFDResidualData(args.data_root, split='test', sensor_down_sample_rate=args.d, prior_path=args.prior)
    gate = {os.path.normpath(os.path.join(args.data_root, f)) for f in json.load(open(SUBSET))['files']}
    val.img_list = [f for f in val.img_list if os.path.normpath(f) not in gate]       # monitoring never sees the gate cubes
    if args.limit_files:
        train.img_list = train.img_list[:args.limit_files]; val.img_list = val.img_list[:max(8, args.limit_files // 8)]
    return train, val


def make_loader(ds, args, batch_size, shuffle, rank, world, device):
    """Shuffled loaders always use a DistributedSampler (num_replicas=1 on one GPU) so the order of epoch e is a pure
    function of (seed, e) via set_epoch — the basis of exact resume."""
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=shuffle, seed=args.seed, drop_last=shuffle) \
        if (shuffle or world > 1) else None
    kw = dict(num_workers=args.num_workers, pin_memory=(device.type == 'cuda'), collate_fn=residual_collate_fn)
    if args.num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)         # safe: d never changes during a run
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=shuffle, **kw)


def make_trainer(args, device):
    return DiffusionTrainer(device=device, n_timesteps=N_TIMESTEPS, loss_type=args.loss_type,
                            prediction_type=args.prediction_type, snr_gamma=args.snr_gamma if args.snr_gamma > 0 else None)


def wrap_fsdp(model, args, dist_ctx):
    mp = None if args.no_bf16 else MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16)
    return FSDP(model, sharding_strategy=getattr(ShardingStrategy, args.sharding_strategy), mixed_precision=mp,
                auto_wrap_policy=functools.partial(size_based_auto_wrap_policy, min_num_params=args.wrap_min_params),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE, device_id=dist_ctx.local_rank,
                sync_module_states=True, use_orig_params=True)


def make_optimizer(model, args):
    return torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8 if args.no_bf16 else 1e-6,
                             weight_decay=args.weight_decay)


def make_scheduler(optimizer, args):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * args.lrf)


# ----------------------------------------------------------------------------- checkpoints
def checkpoint_meta(args, sigma_d, prior_s, prior_sha256):
    """What scripts/eval_warmstart.py::check_meta verifies (mode, d, base_channels, prediction_type, sigma_d, prior_sha256)
    plus provenance."""
    return dict(mode=args.mode, d=args.d, base_channels=args.base_channels, prediction_type=args.prediction_type,
                n_timesteps=N_TIMESTEPS, sigma_d=float(sigma_d), prior_s=float(prior_s), prior_sha256=prior_sha256, args=vars(args))


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, meta, path, elapsed_hours, dist_ctx):
    """EVERY rank must call this (the FSDP gathers and the barrier are collectives); only rank 0 writes, atomically."""
    if dist_ctx.enabled:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                                  FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        optim_state = optimizer.state_dict()
    if dist_ctx.rank == 0:
        ck = dict(epoch=epoch, loss=float(train_loss), model_state_dict=model_state, optimizer_state_dict=optim_state,
                  scheduler_state_dict=scheduler.state_dict(), meta=meta, elapsed_hours=float(elapsed_hours))
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = path + '.tmp'
        try:
            torch.save(ck, tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        print(f'[rank 0] checkpoint saved -> {path}', flush=True)
    if dist_ctx.enabled:
        dist.barrier()


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def restore_training_state(ck, model, optimizer, scheduler, dist_ctx):
    """Model weights are loaded by the caller (before any FSDP wrap). Restores optimizer + scheduler exactly
    (no cosine restart) and returns (start_epoch, elapsed_hours)."""
    osd = ck['optimizer_state_dict']
    if dist_ctx.enabled:
        osd = FSDP.optim_state_dict_to_load(model, optimizer, osd)
    optimizer.load_state_dict(osd)
    scheduler.load_state_dict(ck['scheduler_state_dict'])
    return int(ck['epoch']) + 1, float(ck.get('elapsed_hours', 0.0))


def latest_checkpoint(run_dir):
    cks = [f for f in os.listdir(run_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')] if os.path.isdir(run_dir) else []
    return os.path.join(run_dir, max(cks, key=lambda f: int(f[len('checkpoint_epoch_'):-4]))) if cks else None


def prune_checkpoints(run_dir, keep_last=2, keep_epochs=()):
    cks = sorted((int(f[len('checkpoint_epoch_'):-4]), f) for f in os.listdir(run_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth'))
    keep = {e for e, _ in cks[-keep_last:]} | set(keep_epochs)
    for e, f in cks:
        if e not in keep:
            os.remove(os.path.join(run_dir, f))


def should_save(epoch1, args):
    """epoch1 is 1-based."""
    return epoch1 % args.save_every == 0 or epoch1 in set(args.keep_epochs) or epoch1 == args.num_epochs


# ----------------------------------------------------------------------------- loops
def clip_grads(model, clip, dist_ctx):
    if clip and clip > 0:
        return float(model.clip_grad_norm_(clip)) if dist_ctx.enabled else float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    return float('nan')


def train_one_epoch(model, trainer, loader, optimizer, args, device, dist_ctx, mode, sigma_d, log_prefix=''):
    model.train()
    use_bf16 = not args.no_bf16 and device.type == 'cuda'
    tot, n, skipped, run_skipped, t_data, t0 = 0.0, 0, 0, 0, 0.0, time.time()
    t_fetch = time.time()
    for step, (x, y_d, xh) in enumerate(loader):
        t_data += time.time() - t_fetch
        x, y_d, xh = x.to(device, non_blocking=True), y_d.to(device, non_blocking=True), xh.to(device, non_blocking=True)
        cond = {'sensor': y_d, 'x0_hat': xh}                        # moved per key: get_loss moves only x_0
        target = target_for(mode, x, xh, sigma_d)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            loss, _ = trainer.get_loss(model, target, cond, loss_in_fp32=True)
        if all_min(torch.isfinite(loss).item(), device, dist_ctx) == 0:    # collective: every rank skips together
            skipped += 1; run_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            if run_skipped >= args.max_nonfinite:
                raise RuntimeError(f'{run_skipped} consecutive non-finite losses at step {step}; aborting')
            t_fetch = time.time(); continue
        run_skipped = 0
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grads(model, args.grad_clip, dist_ctx)
        optimizer.step()
        tot += loss.item(); n += 1
        if dist_ctx.rank == 0 and args.log_interval and step % args.log_interval == 0:
            print(f'{log_prefix}step {step}/{len(loader)} loss {loss.item():.4f} lr {optimizer.param_groups[0]["lr"]:.2e}', flush=True)
        t_fetch = time.time()
    agg = torch.tensor([tot, float(n)], device=device)
    if dist_ctx.enabled:
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
    return dict(loss=(agg[0] / max(agg[1], 1)).item(), lr=optimizer.param_groups[0]['lr'], steps=n,
                skipped_nonfinite=skipped, sec=time.time() - t0, data_sec=t_data)


@torch.no_grad()
def validate(model, trainer, loader, args, device, dist_ctx, mode, sigma_d):
    """Loss-only (the training objective) with a fixed RNG so epochs are comparable; never used for selection."""
    model.eval()
    use_bf16 = not args.no_bf16 and device.type == 'cuda'
    tot, n = 0.0, 0
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):            # fork_rng takes CUDA device indices
        torch.manual_seed(args.seed + 12345)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed + 12345)
        for x, y_d, xh in loader:
            x, y_d, xh = x.to(device), y_d.to(device), xh.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                loss, _ = trainer.get_loss(model, target_for(mode, x, xh, sigma_d), {'sensor': y_d, 'x0_hat': xh}, loss_in_fp32=True)
            tot += loss.item() * x.shape[0]; n += x.shape[0]
    agg = torch.tensor([tot, float(n)], device=device)
    if dist_ctx.enabled:
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
    model.train()
    return dict(val_loss=(agg[0] / max(agg[1], 1)).item(), val_n=int(agg[1].item()))


# ----------------------------------------------------------------------------- entry points
def build_everything(args, dist_ctx, device):
    train_ds, val_ds = make_datasets(args)
    model = build_residual_model(args.mode, args.base_channels)
    n_params = sum(p.numel() for p in model.parameters())           # counted before the wrap (sharding hides it after)
    resume_path = latest_checkpoint(args.save_dir) if args.resume == 'auto' else (args.resume or None)
    ck = load_checkpoint(resume_path) if resume_path else None
    if ck is not None:
        model.load_state_dict(ck['model_state_dict'], strict=True)          # before the wrap, on CPU
    model = wrap_fsdp(model, args, dist_ctx) if dist_ctx.enabled else model.to(device)
    trainer = make_trainer(args, device)
    optimizer = make_optimizer(model, args); scheduler = make_scheduler(optimizer, args)
    start_epoch, elapsed = (0, 0.0) if ck is None else restore_training_state(ck, model, optimizer, scheduler, dist_ctx)
    return SimpleNamespace(train_ds=train_ds, val_ds=val_ds, model=model, trainer=trainer, optimizer=optimizer, scheduler=scheduler,
                           start_epoch=start_epoch, elapsed=elapsed, resume_path=resume_path, n_params=n_params)


def run_epochs(args, parts, dist_ctx, device):
    """The epoch loop. `parts` is either build_everything()'s namespace or the tests' tuple (ds, loader, model, trainer, opt, sched)."""
    if isinstance(parts, tuple):
        ds, train_loader, model, trainer, optimizer, scheduler = parts
        val_loader, start_epoch, elapsed, sigma_d = None, 0, 0.0, ds.sigma_d
    else:
        model, trainer, optimizer, scheduler = parts.model, parts.trainer, parts.optimizer, parts.scheduler
        start_epoch, elapsed, sigma_d = parts.start_epoch, parts.elapsed, parts.train_ds.sigma_d
        train_loader = make_loader(parts.train_ds, args, args.batch_size, True, dist_ctx.rank, dist_ctx.world, device)
        val_loader = make_loader(parts.val_ds, args, args.batch_size, False, dist_ctx.rank, dist_ctx.world, device)
    meta = checkpoint_meta(args, sigma_d, getattr(parts, 'train_ds', parts[0] if isinstance(parts, tuple) else None).prior['s'], file_sha256(args.prior) if os.path.exists(args.prior) else '')
    t_start = time.time() - elapsed * 3600                           # before any rank-0-only work: the ranks' clocks agree
    final_epoch = start_epoch
    for epoch in range(start_epoch, args.num_epochs):
        epoch_seed(args, epoch, dist_ctx.rank)
        train_loader.sampler.set_epoch(epoch)
        rec = train_one_epoch(model, trainer, train_loader, optimizer, args, device, dist_ctx, args.mode, sigma_d, log_prefix=f'[epoch {epoch + 1}] ')
        scheduler.step()
        epoch1 = epoch + 1
        if val_loader is not None and args.val_every and epoch1 % args.val_every == 0:
            rec.update(validate(model, trainer, val_loader, args, device, dist_ctx, args.mode, sigma_d))
        hours = (time.time() - t_start) / 3600
        stop = torch.tensor(float(hours >= args.max_hours), device=device)
        if dist_ctx.enabled:
            dist.broadcast(stop, src=0)                               # one decision for every rank
        out_of_time = bool(stop.item())
        rec.update(epoch=epoch1, elapsed_hours=hours, peak_gb=torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else 0.0)
        if should_save(epoch1, args) or out_of_time:
            save_checkpoint(model, optimizer, scheduler, epoch, rec['loss'], meta, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch1}.pth'), hours, dist_ctx)
            if dist_ctx.rank == 0:
                prune_checkpoints(args.save_dir, args.keep_last, list(args.keep_epochs) + [epoch1])
        if dist_ctx.rank == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            with open(os.path.join(args.save_dir, 'train_log.jsonl'), 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f'epoch {epoch1}/{args.num_epochs} loss {rec["loss"]:.5f} lr {rec["lr"]:.2e} '
                  f'{"val " + format(rec["val_loss"], ".5f") + " " if "val_loss" in rec else ""}'
                  f'{rec["sec"]:.0f}s (data {rec["data_sec"]:.0f}s) elapsed {hours:.2f}h', flush=True)
        final_epoch = epoch1
        if out_of_time:
            if dist_ctx.rank == 0:
                print(f'wall-clock budget {args.max_hours} h reached after epoch {epoch1}; stopping', flush=True)
            break
    if dist_ctx.rank == 0:
        with open(os.path.join(args.save_dir, 'DONE'), 'w') as f:
            f.write(f'{final_epoch}\n')
    return final_epoch


def dry_run(args):
    """Time the production step on this configuration. Steady-state data wait excludes the prefetched warm-up batches."""
    dist_ctx, device = setup_dist(args)
    seed_everything(args.seed, dist_ctx.rank)
    parts = build_everything(args, dist_ctx, device)
    loader = make_loader(parts.train_ds, args, args.batch_size, True, dist_ctx.rank, dist_ctx.world, device)
    loader.sampler.set_epoch(0)
    use_bf16 = not args.no_bf16 and device.type == 'cuda'
    warm = max(3, args.num_workers * 4)                               # 4 = prefetch_factor
    it = iter(loader); comp, wall, data = [], [], []
    for step in range(args.dry_run_steps + warm):
        tf = time.time(); x, y_d, xh = next(it); td = time.time() - tf
        x, y_d, xh = x.to(device), y_d.to(device), xh.to(device)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            loss, _ = parts.trainer.get_loss(parts.model, target_for(args.mode, x, xh, parts.train_ds.sigma_d), {'sensor': y_d, 'x0_hat': xh}, loss_in_fp32=True)
        parts.optimizer.zero_grad(set_to_none=True); loss.backward(); clip_grads(parts.model, args.grad_clip, dist_ctx); parts.optimizer.step()
        if device.type == 'cuda': torch.cuda.synchronize(device)
        if step >= warm:
            comp.append(time.time() - t0); data.append(td); wall.append(time.time() - tf)
    world = dist_ctx.world
    rec = dict(config=f'{args.mode}_d{args.d}_b{args.base_channels}_bs{args.batch_size}x{world}_w{args.num_workers}', mode=args.mode, d=args.d,
               base_channels=args.base_channels, batch_size_per_gpu=args.batch_size, world=world, num_workers=args.num_workers, fsdp=dist_ctx.enabled,
               bf16=use_bf16, steps=len(comp), warmup_steps=warm, params=parts.n_params, sec_per_step=float(np.mean(comp)), data_wait_sec=float(np.mean(data)),
               wall_sec_per_step=float(np.mean(wall)), peak_gb=torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else 0.0,
               sec_per_epoch=float(np.mean(wall)) * len(parts.train_ds) / (args.batch_size * world), loss=float(loss.item()), torch=torch.__version__)
    if dist_ctx.rank == 0:
        print(json.dumps(rec, indent=1))
    if dist_ctx.enabled:
        dist.destroy_process_group()
    return rec


def main(args):
    dist_ctx, device = setup_dist(args)
    seed_everything(args.seed, dist_ctx.rank)
    if device.type == 'cuda':
        free = torch.cuda.mem_get_info(device)[0] / 2 ** 20
        if all_min(free, device, dist_ctx) < 3000:                    # collective: every rank refuses together
            raise SystemExit(f'a GPU has less than 3000 MiB free ({device}: {free:.0f}); refusing to start')
    parts = build_everything(args, dist_ctx, device)
    if dist_ctx.rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=1)
        print(f'mode={args.mode} d={args.d} base={args.base_channels} params={parts.n_params:,} sigma_d={parts.train_ds.sigma_d:.6f} '
              f'train={len(parts.train_ds)} val={len(parts.val_ds)} batch/gpu={args.batch_size} world={dist_ctx.world} workers={args.num_workers} '
              f'resume={parts.resume_path} start_epoch={parts.start_epoch} elapsed={parts.elapsed:.2f}h', flush=True)
    run_epochs(args, parts, dist_ctx, device)
    if dist_ctx.enabled:
        dist.destroy_process_group()


if __name__ == '__main__':
    a = get_parser().parse_args()
    if a.dry_run_steps:
        dry_run(a)
    else:
        main(a)
```

Notes for the implementer: the `meta` line in `run_epochs` is deliberately tolerant of the tests' tuple form (the `TinyResidualSet.prior` dict and a non-existent `--prior` path); in production `parts` is the namespace from `build_everything` and `args.prior` exists. `prune_checkpoints` receives `keep_epochs + [epoch1]` so a checkpoint written by the 24 h cap survives the next save. The `DONE` file carries the final epoch the gate must use. On `--resume` the scheduler state (including its `T_max`) comes from the checkpoint, so `--num_epochs` must not change between launches of one run (`launch_run.sh` always passes 100). Under FSDP the `offload_to_cpu=True` state-dict configs are correct on GPUs (a CPU/gloo probe needs `offload_to_cpu=False`; not a production path).

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_train_residual.py -q`
Expected: 16 passed (the parametrised target test counts 3) in about a minute on CPU. Then `CUDA_VISIBLE_DEVICES= python -m pytest tests -q` → 88 passed.

- [ ] **Step 5: Commit**

```bash
git add main_2d_residual_fsdp.py tests/test_train_residual.py
git commit -m "feat(warmstart): residual/standard/standard_xhat training entry point (single GPU or FSDP, collective decisions, atomic meta checkpoints, exact resume)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
```

---

### Task 3: Production-config timing and end-to-end smoke runs (GPU; from the main checkout after merging Tasks 1, 2, 4, 5)

**Files:**
- Create: `scripts/pilot/smoke_test.sh`
- Create (git-ignored): `RESULTS_ROOT/timing_phase1.json`

**Interfaces:**
- Consumes: `main_2d_residual_fsdp.py --dry_run_steps`, `--resume auto`, `scripts/eval_warmstart.py --grid quick`.
- Produces: timing records `residual_d4_b64_bs8x1_w8` (pilot, loader-inclusive), `residual_d4_b256_bs4x2_w8` (full model at batch 4/GPU, FSDP, production wrap), the quick-grid evaluation wall-clock at base 16/64; a verified path training → checkpoint → resume → `check_meta` → eval JSON on one GPU and on two GPUs (FSDP).

- [ ] **Step 1: Check the GPUs**

Run: `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv`
Expected: both GPUs idle (< 200 MiB used). Do not continue otherwise.

- [ ] **Step 2: Time the pilot configuration with data loading included**

Run: `python main_2d_residual_fsdp.py --mode residual --ds 4 --base_channels 64 --batch_size 8 --num_workers 8 --device cuda:0 --save_dir /tmp/unused --dry_run_steps 200`
Expected: `sec_per_step ≈ 0.23`, `data_wait_sec` < 10 % of `sec_per_step` (Review Focus 5), `peak_gb ≈ 3.4`, `sec_per_epoch ≈ 260`. Append the record to `RESULTS_ROOT/timing_phase1.json` (a JSON list). If the data wait exceeds 10 %, repeat with `--num_workers 12` (and 16 if needed), record every attempt, and launch the pilot with the smallest compute-bound worker count.

- [ ] **Step 3: Time the full configuration at batch 4 per GPU (for the 1b decision)**

Run: `torchrun --standalone --nproc_per_node=2 main_2d_residual_fsdp.py --mode residual --ds 4 --base_channels 256 --batch_size 4 --num_workers 8 --save_dir /tmp/unused --dry_run_steps 40`
Expected: finite loss, `peak_gb` well under 16 (Phase 0 measured 16.1 GB at 8/GPU), `sec_per_epoch` for 9000 files at global batch 8 (1125 steps). Append it. Confirm with `nvidia-smi` that no process is left on either GPU.

- [ ] **Step 4: Write and run the smoke test (single GPU and FSDP)**

`scripts/pilot/smoke_test.sh`:

```bash
#!/bin/bash
# End-to-end smoke test on 64 training files: train 2 epochs, resume to 3, quick-grid evaluation, meta refusal.
# Usage: smoke_test.sh MODE GPU            (single GPU)      ~10 min
#        smoke_test.sh MODE fsdp           (both GPUs, FSDP) ~10 min
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python
ROOT=${RESIDUAL_WARMSTART_RESULTS:-/data/chaoyi_he/HSI/Diffu/results/residual_warmstart}
MODE=${1:-residual}; TARGET=${2:-0}
if [[ "$TARGET" == "fsdp" ]]; then
  LAUNCH="$PY -m torch.distributed.run --standalone --nproc_per_node=2"; DEV=""; EVAL_GPU=0; TAG=fsdp; BS=4
else
  LAUNCH="$PY"; DEV="--device cuda:$TARGET"; EVAL_GPU=$TARGET; TAG=gpu$TARGET; BS=8
fi
RUN=$ROOT/runs/smoke_${MODE}_${TAG}_d4_b16
rm -rf "$RUN"
COMMON="main_2d_residual_fsdp.py --mode $MODE --ds 4 --base_channels 16 --batch_size $BS --num_workers 8 $DEV --limit_files 64 --save_every 1 --keep_epochs 1 2 3 --val_every 1 --save_dir $RUN"
$LAUNCH $COMMON --num_epochs 2
[[ -f "$RUN/DONE" ]] || { echo 'no DONE after 2 epochs'; exit 1; }
[[ -f "$RUN/checkpoint_epoch_2.pth" ]] || { echo 'no epoch-2 checkpoint'; exit 1; }
$LAUNCH $COMMON --num_epochs 3 --resume auto                      # resumes from epoch 2, trains epoch 3
[[ "$(cat "$RUN/DONE")" == "3" ]] || { echo 'resume did not reach epoch 3'; exit 1; }
[[ $(wc -l < "$RUN/train_log.jsonl") -eq 3 ]] || { echo 'train_log.jsonl should have 3 lines'; exit 1; }
$PY scripts/eval_warmstart.py --d 4 --ckpt "$RUN/checkpoint_epoch_3.pth" --mode "$MODE" --base_channels 16 --grid quick --seeds 0 --batch_size 8 --device cuda:$EVAL_GPU
JSON=$ROOT/eval/$MODE/d4/smoke_${MODE}_${TAG}_d4_b16__checkpoint_epoch_3__quick.json
[[ -f "$JSON" ]] || { echo "missing $JSON"; exit 1; }
grep -q '"meta_checked": true' "$JSON" || { echo 'meta not checked'; exit 1; }
WRONG=standard; [[ "$MODE" == "standard" ]] && WRONG=residual
if $PY scripts/eval_warmstart.py --d 4 --ckpt "$RUN/checkpoint_epoch_3.pth" --mode $WRONG --base_channels 16 --grid quick --seeds 0 --device cuda:$EVAL_GPU 2>/dev/null; then
  echo 'mode mismatch was NOT refused'; exit 1
fi
echo "smoke test ($MODE, $TAG) passed"
```

Run: `bash scripts/pilot/smoke_test.sh residual 0`, `bash scripts/pilot/smoke_test.sh standard_xhat 1`, then `bash scripts/pilot/smoke_test.sh residual fsdp` (both GPUs must be idle).
Expected: three `smoke test (...) passed` lines; record the wall-clock of each quick-grid evaluation (200 cubes × 1 seed at base 16) in `timing_phase1.json` as `eval_quick_b16_sec`, and scale to the pilot's full grid (base 64, 2 seeds: 828 + 250 NFE per cube-seed vs 48 → ≈ 45 × the quick time) to replace the "≈ 1 h" estimate in Task 6. Delete the `smoke_*` run directories and their eval JSONs afterwards.

- [ ] **Step 5: Commit**

```bash
git add scripts/pilot/smoke_test.sh
git commit -m "feat(warmstart): pilot smoke test (single GPU + FSDP, resume, meta refusal); timing records for the pilot and the batch-4 FSDP configuration

<paste the timing records>

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
```

---

### Task 4: Gate 1a script — paired comparison of three final full-grid evaluations

**Files:**
- Create: `scripts/gate_1a.py`
- Test: `tests/test_gate_1a.py`

**Interfaces:**
- Consumes: eval JSONs from `scripts/eval_warmstart.py` (`grid`, `seeds`, `d`, `provenance.{mode, base_channels, prediction_type, prior_sha256, meta_checked, ckpt}`, `rows[*]` with `name, nfe, rmse_pct, rmse_pct_se, paired_rmse_diff, paired_rmse_se, per_item.rmse_pct [seed][cube], files`, `yswap.passes`, `baseline.rmse_pct`), optional `DONE` files next to the checkpoints.
- Produces: `check_comparable(evs)`, `best_row(ev, max_nfe)`, `paired_vs_model(row_b, row_other)`, `paired_on_intersection(row_x, row_y)`, `gate(ev_b, ev_a, ev_aprime, max_nfe=10, k=2.0, require_done=True)`; CLI `python scripts/gate_1a.py --b <json> --a <json> --aprime <json> [--max_nfe 10] [--k 2] [--no_require_done] [--out <json>]`.

Convention (planner ruling on spec §8 "at the best ≤ 10-NFE setting"): each model is taken at its **own** best eligible row (lowest mean RMSE among rows with `nfe ≤ max_nfe`, excluding `ddpm1000`), marked `selected_from_grid`; `ddim10` is also reported for every model as the pre-registered setting. Paired differences are per cube after averaging seeds, SE over cubes. Verdict matrix: `INVALID` if B's y-swap fails (spec §11); `PASS` if B − baseline, B − A and B − A′ are all < −k·SE; `PASS_NOT_OVER_APRIME` if B beats the baseline and A but not A′ (the gain is attributable to the aligned x̂₀ conditioning rather than the residual parameterisation — the user chooses B or A′ for 1b); `FAIL` otherwise. An ablation whose own y-swap fails is flagged `ablation_blind` in its comparison (reported, not fatal). The gate refuses JSONs that are not comparable final full-grid evaluations.

- [ ] **Step 1: Write the failing tests**

`tests/test_gate_1a.py`:

```python
import numpy as np
import pytest

from scripts.gate_1a import best_row, check_comparable, gate, paired_on_intersection, paired_vs_model

FILES = [f'f{i}' for i in range(6)]
PROV = dict(d=4, base_channels=64, prediction_type='v', prior_sha256='p' * 64, meta_checked=True, ckpt='/x/runs/r/checkpoint_epoch_100.pth')


def row(name, nfe, vals, base=None, files=FILES):
    vals = np.asarray(vals, dtype=float)                     # [seed][cube]
    cube = vals.mean(0)
    base = np.asarray(base if base is not None else cube + 0.5)
    diff = cube - base
    return dict(name=name, nfe=nfe, rmse_pct=float(cube.mean()), rmse_pct_se=float(cube.std(ddof=1) / np.sqrt(len(cube))),
                paired_rmse_diff=float(diff.mean()), paired_rmse_se=float(diff.std(ddof=1) / np.sqrt(len(diff))),
                per_item={'rmse_pct': vals.tolist()}, files=list(files))


def ev(mode, rows, passes=True, **prov):
    return dict(provenance=dict(PROV, mode=mode, **prov), grid='full', seeds=[0, 1], d=4, rows=rows, yswap=dict(passes=passes), baseline=dict(rmse_pct=2.35))


def test_best_row_respects_nfe_and_excludes_ddpm1000():
    rows = [row('ddpm1000', 1000, [[0.1] * 6]), row('ddim20', 20, [[0.2] * 6]), row('ddim10', 10, [[0.5] * 6]), row('warm100_ddim', 10, [[0.4] * 6])]
    assert best_row(ev('residual', rows), 10)['name'] == 'warm100_ddim'
    assert best_row(ev('residual', rows), 20)['name'] == 'ddim20'
    assert best_row(ev('residual', rows), 1000)['name'] == 'ddim20'      # ddpm1000 is never eligible


def test_paired_vs_model_uses_per_cube_differences_and_same_cubes():
    b = row('ddim10', 10, [[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]] * 2)
    a = row('ddim5', 5, [[2.0, 2.1, 2.2, 2.3, 2.4, 2.5]] * 2)
    r = paired_vs_model(b, a)
    assert r['diff'] == pytest.approx(-1.0) and r['se'] == pytest.approx(0.0, abs=1e-12) and r['n'] == 6
    with pytest.raises(ValueError):
        paired_vs_model(b, row('ddim10', 10, [[1.0] * 6], files=FILES[::-1]))


def test_paired_on_intersection_pairs_by_file():
    x = row('ddim10', 10, [[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]] * 2)
    y = row('ddpm1000', 1000, [[0.9, 1.0, 1.3]] * 2, files=['f0', 'f1', 'f5'])
    r = paired_on_intersection(x, y)
    assert r['n'] == 3 and r['diff'] == pytest.approx(np.mean([0.1, 0.1, 0.2]))


def test_gate_threshold_uses_k_paired_se():
    base = np.full(6, 2.3)
    a_cube = np.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0]); b_cube = a_cube + np.array([-0.3, 0.1, -0.2, 0.0, -0.25, 0.05])
    b = ev('residual', [row('ddim10', 10, [b_cube, b_cube], base)])
    a = ev('standard', [row('ddim10', 10, [a_cube, a_cube], base)])
    ap = ev('standard_xhat', [row('ddim10', 10, [a_cube + 0.5, a_cube + 0.5], base)])
    g2, g1 = gate(b, a, ap, k=2.0, require_done=False), gate(b, a, ap, k=1.0, require_done=False)
    assert g2['vs_a']['diff'] == pytest.approx(-0.1) and g2['vs_a']['se'] == pytest.approx(0.0695, abs=1e-3)
    assert g2['vs_a']['pass'] is False and g2['verdict'] == 'FAIL'
    assert g1['vs_a']['pass'] is True


def test_gate_verdict_matrix():
    base = np.full(6, 2.3); n = np.linspace(-0.05, 0.05, 6)
    b = ev('residual', [row('ddim10', 10, [1.8 + n, 1.9 + n], base)])
    a = ev('standard', [row('ddim10', 10, [2.6 + n, 2.7 + n], base)])
    ap = ev('standard_xhat', [row('ddim10', 10, [2.4 + n, 2.5 + n], base)])
    g = gate(b, a, ap, require_done=False)
    assert g['verdict'] == 'PASS' and g['b']['selected_from_grid'] == 'ddim10' and g['prereg']['setting'] == 'ddim10'
    ap_equal = ev('standard_xhat', [row('ddim10', 10, [1.8 + n[::-1], 1.9 + n[::-1]], base)])   # same mean as B, per-cube spread
    assert gate(b, a, ap_equal, require_done=False)['verdict'] == 'PASS_NOT_OVER_APRIME'
    a_better = ev('standard', [row('ddim10', 10, [1.7 + n, 1.7 + n], base)])
    assert gate(b, a_better, ap, require_done=False)['verdict'] == 'FAIL'
    assert gate(ev('residual', b['rows'], passes=False), a, ap, require_done=False)['verdict'] == 'INVALID'
    assert gate(b, ev('standard', a['rows'], passes=False), ap, require_done=False)['vs_a']['ablation_blind'] is True


def test_gate_refuses_incomparable_inputs(tmp_path):
    base = np.full(6, 2.3)
    b = ev('residual', [row('ddim10', 10, [[1.8] * 6] * 2, base)]); a = ev('standard', [row('ddim10', 10, [[2.6] * 6] * 2, base)])
    ap = ev('standard_xhat', [row('ddim10', 10, [[2.4] * 6] * 2, base)])
    with pytest.raises(ValueError, match='mode'):
        gate(a, b, ap, require_done=False)
    for bad in (dict(a, grid='quick'), dict(a, seeds=[0]), dict(a, d=2), ev('standard', a['rows'], base_channels=256),
                ev('standard', a['rows'], prior_sha256='q' * 64), ev('standard', a['rows'], meta_checked=False)):
        with pytest.raises(ValueError):
            check_comparable([b, bad, ap])
    run = tmp_path / 'r'; run.mkdir(); (run / 'DONE').write_text('100\n')
    ok = ev('residual', b['rows'], ckpt=str(run / 'checkpoint_epoch_100.pth'))
    stale = ev('residual', b['rows'], ckpt=str(run / 'checkpoint_epoch_50.pth'))
    assert gate(ok, a, ap, require_done=False)['verdict'] in ('PASS', 'PASS_NOT_OVER_APRIME', 'FAIL')
    with pytest.raises(ValueError, match='DONE'):
        gate(stale, a, ap, require_done=True)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_gate_1a.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.gate_1a'`.

- [ ] **Step 3: Write the script**

`scripts/gate_1a.py`:

```python
"""Phase 1a gate (spec §8): at its own best <= 10-NFE sampler setting, B must beat the baseline, A and A' by more
than k paired standard errors on the same cubes; a failing y-swap makes B invalid (spec §11). Refuses anything that
is not a set of three comparable final full-grid evaluations.

python scripts/gate_1a.py --b <eval>/residual/d4/residual_d4_b64__checkpoint_epoch_100.json \
    --a <eval>/standard/d4/standard_d4_b64__checkpoint_epoch_100.json \
    --aprime <eval>/standard_xhat/d4/standard_xhat_d4_b64__checkpoint_epoch_100.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

EXPECTED = {'b': 'residual', 'a': 'standard', 'aprime': 'standard_xhat'}
PREREG = 'ddim10'


def check_comparable(evs):
    """All three must be full-grid, meta-checked evaluations with identical seeds, d, base_channels, prediction_type and prior."""
    for e in evs:
        if e.get('grid', 'full') != 'full':
            raise ValueError(f"not a full-grid evaluation: grid={e.get('grid')!r}")
        if not e['provenance'].get('meta_checked'):
            raise ValueError('checkpoint meta was not checked (meta_checked is false)')
    for key in ('seeds', 'd'):
        if len({json.dumps(e[key]) for e in evs}) != 1:
            raise ValueError(f'evaluations differ in {key}: {[e[key] for e in evs]}')
    for key in ('base_channels', 'prediction_type', 'prior_sha256'):
        if len({str(e['provenance'][key]) for e in evs}) != 1:
            raise ValueError(f'evaluations differ in provenance.{key}: {[e["provenance"][key] for e in evs]}')


def check_done(ev):
    """The checkpoint must be the run's final one: its epoch equals the DONE file next to it."""
    ckpt = ev['provenance']['ckpt']
    done = os.path.join(os.path.dirname(ckpt), 'DONE')
    if not os.path.exists(done):
        raise ValueError(f'no DONE file next to {ckpt}: the run has not finished')
    m = re.search(r'checkpoint_epoch_(\d+)\.pth$', ckpt)
    final = int(open(done).read().strip())
    if not m or int(m.group(1)) != final:
        raise ValueError(f'{ckpt} is not the final checkpoint (DONE says epoch {final})')


def eligible_rows(ev, max_nfe):
    return [r for r in ev['rows'] if r['name'] != 'ddpm1000' and r['nfe'] <= max_nfe]


def best_row(ev, max_nfe):
    rows = eligible_rows(ev, max_nfe)
    if not rows:
        raise ValueError(f'no sampler row with nfe <= {max_nfe}')
    return min(rows, key=lambda r: r['rmse_pct'])


def per_cube(row):
    return np.asarray(row['per_item']['rmse_pct'], dtype=float).mean(0)       # [seed][cube] -> [cube]


def _paired(d):
    return dict(diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan'), n=int(len(d)))


def paired_vs_model(row_b, row_other):
    if row_b['files'] != row_other['files']:
        raise ValueError('rows score different cubes (or a different order); pair via files')
    return _paired(per_cube(row_b) - per_cube(row_other))


def paired_on_intersection(row_x, row_y):
    """x − y on the cubes both rows scored (e.g. DDIM-10 on 200 cubes vs DDPM-1000 on 50)."""
    cx = dict(zip(row_x['files'], per_cube(row_x))); cy = dict(zip(row_y['files'], per_cube(row_y)))
    common = [f for f in row_y['files'] if f in cx]
    if not common:
        raise ValueError('no common cubes')
    return _paired(np.array([cx[f] - cy[f] for f in common]))


def row_named(ev, name):
    return next((r for r in ev['rows'] if r['name'] == name), None)


def gate(ev_b, ev_a, ev_aprime, max_nfe=10, k=2.0, require_done=True):
    evs = dict(b=ev_b, a=ev_a, aprime=ev_aprime)
    for slot, ev in evs.items():
        if ev['provenance']['mode'] != EXPECTED[slot]:
            raise ValueError(f"--{slot} must be a {EXPECTED[slot]} evaluation, got mode {ev['provenance']['mode']!r}")
    check_comparable(list(evs.values()))
    if require_done:
        for ev in evs.values():
            check_done(ev)
    rb, ra, rp = (best_row(ev, max_nfe) for ev in (ev_b, ev_a, ev_aprime))
    vs = dict(vs_baseline=dict(diff=rb['paired_rmse_diff'], se=rb['paired_rmse_se'], n=len(rb['files'])),
              vs_a=paired_vs_model(rb, ra), vs_aprime=paired_vs_model(rb, rp))
    for key, r in vs.items():
        r['pass'] = bool(r['diff'] + k * r['se'] < 0)
    vs['vs_a']['ablation_blind'] = not bool(ev_a['yswap']['passes'])
    vs['vs_aprime']['ablation_blind'] = not bool(ev_aprime['yswap']['passes'])
    yswap_b = bool(ev_b['yswap']['passes'])
    if not yswap_b:
        verdict = 'INVALID'
    elif vs['vs_baseline']['pass'] and vs['vs_a']['pass']:
        verdict = 'PASS' if vs['vs_aprime']['pass'] else 'PASS_NOT_OVER_APRIME'
    else:
        verdict = 'FAIL'
    prereg = {s: dict(rmse_pct=row_named(ev, PREREG)['rmse_pct'], paired_rmse_diff=row_named(ev, PREREG)['paired_rmse_diff'])
              for s, ev in evs.items() if row_named(ev, PREREG)}
    at_b_setting = {s: paired_vs_model(rb, row_named(ev, rb['name'])) for s, ev in (('a', ev_a), ('aprime', ev_aprime)) if row_named(ev, rb['name'])}
    speed = {s: paired_on_intersection(row_named(ev, PREREG), row_named(ev, 'ddpm1000'))
             for s, ev in evs.items() if row_named(ev, PREREG) and row_named(ev, 'ddpm1000')}
    return dict(verdict=verdict, max_nfe=max_nfe, k=k, yswap_b=yswap_b,
                b=dict(selected_from_grid=rb['name'], nfe=rb['nfe'], rmse_pct=rb['rmse_pct'], rmse_pct_se=rb['rmse_pct_se'], ckpt=ev_b['provenance']['ckpt']),
                a=dict(selected_from_grid=ra['name'], nfe=ra['nfe'], rmse_pct=ra['rmse_pct'], ckpt=ev_a['provenance']['ckpt']),
                aprime=dict(selected_from_grid=rp['name'], nfe=rp['nfe'], rmse_pct=rp['rmse_pct'], ckpt=ev_aprime['provenance']['ckpt']),
                baseline_rmse_pct=ev_b['baseline']['rmse_pct'], prereg=dict(setting=PREREG, **prereg) if prereg else {},
                b_setting_applied_to=at_b_setting, ddim10_minus_ddpm1000=speed, **vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', required=True); ap.add_argument('--a', required=True); ap.add_argument('--aprime', required=True)
    ap.add_argument('--max_nfe', type=int, default=10); ap.add_argument('--k', type=float, default=2.0)
    ap.add_argument('--no_require_done', action='store_true'); ap.add_argument('--out', default=None)
    args = ap.parse_args()
    evs = [json.load(open(p)) for p in (args.b, args.a, args.aprime)]
    g = gate(*evs, max_nfe=args.max_nfe, k=args.k, require_done=not args.no_require_done)
    print(f"gate 1a (<= {g['max_nfe']} NFE, k = {g['k']}): {g['verdict']}")
    for lab, key in (('B ', 'b'), ('A ', 'a'), ("A'", 'aprime')):
        r = g[key]
        print(f"  {lab} {r['selected_from_grid']:>13} nfe={r['nfe']:3d} RMSE {r['rmse_pct']:.4f} %   {os.path.basename(r['ckpt'])}")
    print(f"  baseline (200 cubes) {g['baseline_rmse_pct']:.4f} %   y-swap B {'pass' if g['yswap_b'] else 'FAIL'}")
    for lab, key in (('B - baseline', 'vs_baseline'), ('B - A', 'vs_a'), ("B - A'", 'vs_aprime')):
        r = g[key]
        flag = '  [ablation y-swap failed]' if r.get('ablation_blind') else ''
        print(f"  {lab:<13} {r['diff']:+.4f} ± {r['se']:.4f} (n={r['n']})  {'pass' if r['pass'] else 'fail'}{flag}")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.b)), 'gate_1a.json')
    with open(out, 'w') as f:
        json.dump(g, f, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_gate_1a.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/gate_1a.py tests/test_gate_1a.py
git commit -m "feat(warmstart): gate 1a — paired B vs baseline / A / A' at each model's best <= 10-NFE setting, with comparability and final-checkpoint checks

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
```

---

### Task 5: Pilot driver (idempotent state machine), launch/eval helpers, status

**Files:**
- Create: `scripts/pilot/launch_run.sh`, `scripts/pilot/eval_milestone.sh`, `scripts/pilot/pilot_driver.py`, `scripts/pilot/status.py`
- Test: `tests/test_pilot_driver.py`

**Interfaces:**
- Consumes: `main_2d_residual_fsdp.py`, `scripts/eval_warmstart.py`, `scripts/gate_1a.py`.
- Produces: `launch_run.sh MODE GPU` (background run under `RESULTS_ROOT/runs/<MODE>_d4_b64/` with `run.log`, `pid`; refuses a busy GPU; adds `--resume auto` when checkpoints exist); `eval_milestone.sh MODE EPOCH quick|full GPU` (full requires an idle GPU and records an `nvidia-smi` snapshot); `pilot_driver.py [--once | --loop SECONDS]` with pure functions `run_state(run_dir, eval_dir) -> dict` and `next_actions(states, gpu_status) -> list[tuple]`, an event log `RESULTS_ROOT/runs/pilot_events.log`; `status.py` (one line per run).

Driver policy (pure `next_actions`): B → GPU 0, A → GPU 1 at start; a run with a dead pid and no `DONE` is relaunched (resume) on its GPU; A′ → GPU 1 when A is `DONE` and GPU 1 is free; quick evaluation of epoch 25/50 on the run's own GPU as soon as the checkpoint exists and the quick JSON does not; full evaluation of the final checkpoint on an idle GPU once the run is `DONE` (one evaluation at a time per GPU); gate when the three full JSONs exist and `gate_1a.json` does not. Every action is appended to the event log with a timestamp; re-invoking the driver after any interruption continues from the observed state. Actions run synchronously, so a run that dies during a 1–3 h full-grid evaluation is relaunched on the pass after it; a run that has been relaunched `MAX_RELAUNCHES` (3) times is left alone and shows up in `status.py` as dead without `DONE` — a human decides.

- [ ] **Step 1: Write the failing tests**

`tests/test_pilot_driver.py`:

```python
import json
import os

from scripts.pilot.pilot_driver import next_actions, run_state

MODES = ('residual', 'standard', 'standard_xhat')


def mk_run(root, mode, epochs=(), done=None, alive=False, log_epochs=0):
    d = root / 'runs' / f'{mode}_d4_b64'; d.mkdir(parents=True, exist_ok=True)
    for e in epochs:
        (d / f'checkpoint_epoch_{e}.pth').write_bytes(b'x')
    if done is not None:
        (d / 'DONE').write_text(f'{done}\n')
    if alive:
        (d / 'pid').write_text(str(os.getpid()))             # this process: alive
    with open(d / 'train_log.jsonl', 'w') as f:
        for e in range(1, log_epochs + 1):
            f.write(json.dumps(dict(epoch=e, loss=1.0, lr=1e-4, sec=10, data_sec=1, elapsed_hours=0.1 * e, skipped_nonfinite=0)) + '\n')
    return str(d)


def mk_eval(root, mode, run, epoch, grid='full'):
    d = root / 'eval' / mode / 'd4'; d.mkdir(parents=True, exist_ok=True)
    (d / f'{run}__checkpoint_epoch_{epoch}{"" if grid == "full" else "__quick"}.json').write_text('{}')


def states(root):
    return {m: run_state(str(root / 'runs' / f'{m}_d4_b64'), str(root / 'eval')) for m in MODES}


def test_run_state_reads_markers(tmp_path):
    d = mk_run(tmp_path, 'residual', epochs=(10, 20, 25), alive=True, log_epochs=27)
    mk_eval(tmp_path, 'residual', 'residual_d4_b64', 25, 'quick')
    s = run_state(d, str(tmp_path / 'eval'))
    assert s['exists'] and s['alive'] and s['done'] is None and s['last_epoch'] == 27
    assert s['checkpoints'] == [10, 20, 25] and s['quick_evals'] == [25] and s['full_evals'] == []
    assert run_state(str(tmp_path / 'runs' / 'standard_d4_b64'), str(tmp_path / 'eval'))['exists'] is False


def test_initial_launch_assigns_b_and_a(tmp_path):
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'free'})
    assert ('launch', 'residual', 0) in acts and ('launch', 'standard', 1) in acts and not any(a[1] == 'standard_xhat' for a in acts)


def test_dead_run_is_relaunched_and_busy_gpu_waits(tmp_path):
    mk_run(tmp_path, 'residual', epochs=(10,), alive=False, log_epochs=12)
    mk_run(tmp_path, 'standard', alive=True, log_epochs=3)
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'busy'})
    assert ('launch', 'residual', 0) in acts and not any(a[1] == 'standard' for a in acts)
    (tmp_path / 'runs' / 'residual_d4_b64' / 'relaunches').write_text('3\n')              # crash loop: left for a human
    assert not any(a[0] == 'launch' and a[1] == 'residual' for a in next_actions(states(tmp_path), {0: 'free', 1: 'busy'}))


def test_run_state_mode_parsing_for_other_run_names(tmp_path):
    d = tmp_path / 'runs' / 'residual_d4_b256'; d.mkdir(parents=True)
    (tmp_path / 'eval' / 'residual' / 'd4').mkdir(parents=True)
    (tmp_path / 'eval' / 'residual' / 'd4' / 'residual_d4_b256__checkpoint_epoch_200.json').write_text('{}')
    s = run_state(str(d), str(tmp_path / 'eval'))
    assert s['mode'] == 'residual' and s['full_evals'] == [200]


def test_quick_eval_when_milestone_present_and_json_absent(tmp_path):
    mk_run(tmp_path, 'residual', epochs=(20, 25), alive=True, log_epochs=26)
    mk_run(tmp_path, 'standard', epochs=(20, 25), alive=True, log_epochs=26)
    mk_eval(tmp_path, 'standard', 'standard_d4_b64', 25, 'quick')
    acts = next_actions(states(tmp_path), {0: 'busy', 1: 'busy'})
    assert ('eval_quick', 'residual', 25, 0) in acts and not any(a[0] == 'eval_quick' and a[1] == 'standard' for a in acts)


def test_aprime_starts_when_a_done_and_full_evals_need_idle_gpu(tmp_path):
    mk_run(tmp_path, 'residual', epochs=(90, 100, 25, 50), alive=True, log_epochs=99)
    mk_run(tmp_path, 'standard', epochs=(90, 100, 25, 50), done=100, alive=False, log_epochs=100)
    acts = next_actions(states(tmp_path), {0: 'busy', 1: 'free'})
    assert ('launch', 'standard_xhat', 1) in acts
    assert not any(a[0] == 'eval_full' for a in acts)                 # GPU 1 goes to A′ first; A's full eval waits for an idle GPU
    mk_run(tmp_path, 'standard_xhat', alive=True, log_epochs=1)
    mk_run(tmp_path, 'residual', epochs=(90, 100, 25, 50), done=100, alive=False, log_epochs=100)
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'busy'})
    assert acts and acts[0][:3] == ('eval_full', 'residual', 100) and acts[0][3] == 0 and len([a for a in acts if a[0] == 'eval_full']) == 1


def test_gate_when_three_full_evals_exist(tmp_path):
    for m in MODES:
        mk_run(tmp_path, m, epochs=(100,), done=100, log_epochs=100)
        mk_eval(tmp_path, m, f'{m}_d4_b64', 100)
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'free'})
    assert acts == [('gate', 100, 100, 100)]
    (tmp_path / 'eval' / 'residual' / 'd4' / 'gate_1a.json').write_text('{}')
    assert next_actions(states(tmp_path), {0: 'free', 1: 'free'}) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_pilot_driver.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.pilot'` (create `scripts/pilot/__init__.py`).

- [ ] **Step 3: Write the scripts**

`scripts/pilot/launch_run.sh`:

```bash
#!/bin/bash
# Launch one pilot run in the background on one GPU. Usage: launch_run.sh residual|standard|standard_xhat GPU [extra args]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python
ROOT=${RESIDUAL_WARMSTART_RESULTS:-/data/chaoyi_he/HSI/Diffu/results/residual_warmstart}
MODE=$1; GPU=$2; shift 2
RUN=$ROOT/runs/${MODE}_d4_b64
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU")
PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id="$GPU" | sed '/^$/d' | wc -l)
if [[ "$FREE" -lt 3000 || "$PROCS" -gt 0 ]]; then echo "GPU $GPU busy (free ${FREE} MiB, ${PROCS} procs); not launching"; exit 1; fi
mkdir -p "$RUN"
RESUME=""
if compgen -G "$RUN/checkpoint_epoch_*.pth" > /dev/null; then
  RESUME="--resume auto"
  echo $(( $(cat "$RUN/relaunches" 2>/dev/null || echo 0) + 1 )) > "$RUN/relaunches"     # the driver stops after MAX_RELAUNCHES
fi
nohup $PY main_2d_residual_fsdp.py --mode "$MODE" --ds 4 --base_channels 64 --batch_size 8 --num_workers 8 --device "cuda:$GPU" \
    --num_epochs 100 --max_hours 24 --save_every 10 --keep_epochs 25 50 100 --val_every 5 --save_dir "$RUN" $RESUME "$@" \
    >> "$RUN/run.log" 2>&1 &
echo $! > "$RUN/pid"
echo "launched $MODE on GPU $GPU (pid $(cat "$RUN/pid"), resume='${RESUME}') -> $RUN/run.log"
```

`scripts/pilot/eval_milestone.sh`:

```bash
#!/bin/bash
# Evaluate one pilot checkpoint. Usage: eval_milestone.sh MODE EPOCH quick|full GPU   (full requires an idle GPU)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python
ROOT=${RESIDUAL_WARMSTART_RESULTS:-/data/chaoyi_he/HSI/Diffu/results/residual_warmstart}
MODE=$1; EPOCH=$2; GRID=$3; GPU=$4
RUN=$ROOT/runs/${MODE}_d4_b64
CK=$RUN/checkpoint_epoch_${EPOCH}.pth
[[ -f "$CK" ]] || { echo "missing $CK"; exit 1; }
LOG=$RUN/eval_${GRID}_epoch_${EPOCH}.log
if [[ "$GRID" == "full" ]]; then
  PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id="$GPU" | sed '/^$/d' | wc -l)
  [[ "$PROCS" -eq 0 ]] || { echo "GPU $GPU busy; the full-grid evaluation needs an idle GPU"; exit 1; }
  nvidia-smi --id="$GPU" >> "$LOG"
fi
$PY scripts/eval_warmstart.py --d 4 --ckpt "$CK" --mode "$MODE" --base_channels 64 --grid "$GRID" --seeds 0 1 --batch_size 8 --device "cuda:$GPU" 2>&1 | tee -a "$LOG"
```

`scripts/pilot/__init__.py`: empty.

`scripts/pilot/pilot_driver.py`:

```python
"""Idempotent driver for the Phase 1a pilot: observe the run directories, take the next actions, log them.
    python scripts/pilot/pilot_driver.py --once          # one pass
    python scripts/pilot/pilot_driver.py --loop 600      # every 10 min until the gate has run (nohup this)
Policy: B -> GPU 0 and A -> GPU 1; a dead run without DONE is relaunched with --resume auto; A' -> GPU 1 after A is DONE;
quick evals of epochs 25/50 on the run's own GPU; full eval of the final checkpoint on an idle GPU; gate when all three exist.
"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
ROOT = os.environ.get('RESIDUAL_WARMSTART_RESULTS', '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart')
PY = '/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python'
MODES = ('residual', 'standard', 'standard_xhat')
HOME_GPU = {'residual': 0, 'standard': 1, 'standard_xhat': 1}
MILESTONES = (25, 50)
MAX_RELAUNCHES = 3            # a run that keeps dying (e.g. the non-finite abort) is left for a human after 3 relaunches


def run_name(mode):
    return f'{mode}_d4_b64'


def pid_alive(path):
    try:
        pid = int(open(path).read().strip()); os.kill(pid, 0); return True
    except (OSError, ValueError):
        return False


def run_state(run_dir, eval_dir):
    """Everything the policy needs, read from the filesystem."""
    name = os.path.basename(run_dir)
    mode = re.sub(r'_d\d+_b\d+$', '', name)                            # works for *_d4_b64, *_d4_b256 and smoke runs
    if not os.path.isdir(run_dir):
        return dict(mode=mode, exists=False, alive=False, done=None, last_epoch=0, checkpoints=[], quick_evals=[], full_evals=[], relaunches=0)
    cks = sorted(int(m.group(1)) for f in os.listdir(run_dir) for m in [re.match(r'checkpoint_epoch_(\d+)\.pth$', f)] if m)
    done_f = os.path.join(run_dir, 'DONE')
    done = int(open(done_f).read().strip()) if os.path.exists(done_f) else None
    last = 0
    log = os.path.join(run_dir, 'train_log.jsonl')
    if os.path.exists(log):
        for line in open(log):
            try:
                last = max(last, json.loads(line)['epoch'])
            except (ValueError, KeyError):
                pass                                                  # a half-written last line
    evs = glob.glob(os.path.join(eval_dir, mode, 'd4', f'{name}__checkpoint_epoch_*.json'))
    quick = sorted(int(m.group(1)) for p in evs for m in [re.search(r'checkpoint_epoch_(\d+)__quick\.json$', p)] if m)
    full = sorted(int(m.group(1)) for p in evs for m in [re.search(r'checkpoint_epoch_(\d+)\.json$', p)] if m)
    rl = os.path.join(run_dir, 'relaunches')
    relaunches = int(open(rl).read().strip() or 0) if os.path.exists(rl) else 0
    return dict(mode=mode, exists=True, alive=pid_alive(os.path.join(run_dir, 'pid')), done=done, last_epoch=last,
                checkpoints=cks, quick_evals=quick, full_evals=full, relaunches=relaunches,
                gate=os.path.exists(os.path.join(eval_dir, 'residual', 'd4', 'gate_1a.json')))


def next_actions(states, gpu_status):
    """Pure policy. states: {mode: run_state}; gpu_status: {0: 'free'|'busy', 1: ...}. Returns a list of action tuples."""
    acts, free = [], {g for g, s in gpu_status.items() if s == 'free'}
    s = states
    if all(s[m]['done'] is not None and s[m]['done'] in s[m]['full_evals'] for m in MODES):
        return [] if s['residual'].get('gate') else [('gate',) + tuple(s[m]['done'] for m in MODES)]
    def launchable(st):                                                # not finished, not running, not stuck in a crash loop
        return st['done'] is None and not st['alive'] and st.get('relaunches', 0) < MAX_RELAUNCHES
    for mode in ('residual', 'standard'):                              # launch or relaunch B and A on their home GPUs
        st, g = s[mode], HOME_GPU[mode]
        if launchable(st) and g in free:
            acts.append(('launch', mode, g)); free.discard(g)
    st = s['standard_xhat']
    if s['standard']['done'] is not None and launchable(st) and HOME_GPU['standard_xhat'] in free:
        acts.append(('launch', 'standard_xhat', HOME_GPU['standard_xhat'])); free.discard(HOME_GPU['standard_xhat'])
    for mode in MODES:                                                 # quick milestone evals on the run's own GPU
        st = s[mode]
        for e in MILESTONES:
            if e in st['checkpoints'] and e not in st['quick_evals'] and st['done'] is None:
                acts.append(('eval_quick', mode, e, HOME_GPU[mode]))
    for mode in MODES:                                                 # one full eval per idle GPU
        st = s[mode]
        if st['done'] is not None and st['done'] not in st['full_evals'] and free:
            g = min(free); acts.append(('eval_full', mode, st['done'], g)); free.discard(g)
    return acts


def gpu_status():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], capture_output=True, text=True).stdout
    uuids = subprocess.run(['nvidia-smi', '--query-gpu=index,gpu_uuid', '--format=csv,noheader'], capture_output=True, text=True).stdout
    idx = {u.strip(): int(i) for i, u in (l.split(',') for l in uuids.strip().splitlines())}
    busy = {idx[l.split(',')[0].strip()] for l in out.strip().splitlines() if l.strip()}
    return {i: ('busy' if i in busy else 'free') for i in idx.values()}


def log_event(msg):
    os.makedirs(os.path.join(ROOT, 'runs'), exist_ok=True)
    line = f'{datetime.datetime.now().isoformat(timespec="seconds")} {msg}'
    print(line, flush=True)
    with open(os.path.join(ROOT, 'runs', 'pilot_events.log'), 'a') as f:
        f.write(line + '\n')


def execute(action):
    kind = action[0]
    sh = os.path.join(REPO, 'scripts', 'pilot')
    if kind == 'launch':
        cmd = ['bash', os.path.join(sh, 'launch_run.sh'), action[1], str(action[2])]
    elif kind in ('eval_quick', 'eval_full'):
        cmd = ['bash', os.path.join(sh, 'eval_milestone.sh'), action[1], str(action[2]), kind[5:], str(action[3])]
    elif kind == 'gate':
        ev = os.path.join(ROOT, 'eval')
        cmd = [PY, os.path.join(REPO, 'scripts', 'gate_1a.py'),
               '--b', os.path.join(ev, 'residual', 'd4', f'residual_d4_b64__checkpoint_epoch_{action[1]}.json'),
               '--a', os.path.join(ev, 'standard', 'd4', f'standard_d4_b64__checkpoint_epoch_{action[2]}.json'),
               '--aprime', os.path.join(ev, 'standard_xhat', 'd4', f'standard_xhat_d4_b64__checkpoint_epoch_{action[3]}.json')]
    else:
        raise ValueError(action)
    log_event(f'START {action}')
    r = subprocess.run(cmd, cwd=REPO)
    log_event(f'END   {action} rc={r.returncode}')


def one_pass():
    states = {m: run_state(os.path.join(ROOT, 'runs', run_name(m)), os.path.join(ROOT, 'eval')) for m in MODES}
    acts = next_actions(states, gpu_status())
    for a in acts:
        execute(a)
    return states, acts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--once', action='store_true'); ap.add_argument('--loop', type=int, default=0)
    args = ap.parse_args()
    while True:
        states, acts = one_pass()
        if states['residual'].get('gate') or not args.loop:
            break
        time.sleep(args.loop)


if __name__ == '__main__':
    main()
```

`scripts/pilot/status.py`:

```python
"""One line per run: last epoch, loss, val, lr, elapsed, ETA, alive/DONE, evaluations present."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.pilot.pilot_driver import ROOT, run_state  # noqa: E402

for run in sorted(glob.glob(os.path.join(ROOT, 'runs', '*_d4_b*'))):
    st = run_state(run, os.path.join(ROOT, 'eval'))
    recs = []
    log = os.path.join(run, 'train_log.jsonl')
    if os.path.exists(log):
        for line in open(log):
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    name = os.path.basename(run)
    if not recs:
        print(f'{name:<24} no epochs yet   alive={st["alive"]} DONE={st["done"]}'); continue
    r = recs[-1]
    n = json.load(open(os.path.join(run, 'args.json')))['num_epochs'] if os.path.exists(os.path.join(run, 'args.json')) else 100
    print(f"{name:<24} epoch {r['epoch']:3d}/{n} loss {r['loss']:.4f} val {r.get('val_loss', float('nan')):.4f} lr {r['lr']:.1e} "
          f"{r['sec']:.0f}s/ep (data {100 * r['data_sec'] / max(r['sec'], 1e-9):.0f}%) elapsed {r['elapsed_hours']:.1f}h "
          f"eta {(n - r['epoch']) * r['sec'] / 3600:.1f}h skipped {r['skipped_nonfinite']} alive={st['alive']} DONE={st['done']} "
          f"quick={st['quick_evals']} full={st['full_evals']}")
```

- [ ] **Step 4: Run the tests and dry-check the shell scripts**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_pilot_driver.py -q && bash -n scripts/pilot/launch_run.sh && bash -n scripts/pilot/eval_milestone.sh && bash -n scripts/pilot/smoke_test.sh`
Expected: 8 passed; no syntax errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/pilot/__init__.py scripts/pilot/launch_run.sh scripts/pilot/eval_milestone.sh scripts/pilot/pilot_driver.py scripts/pilot/status.py tests/test_pilot_driver.py
git commit -m "feat(warmstart): idempotent pilot driver, launch/eval helpers and status

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
```

---

### Task 6: Run the pilot, gate, report (GPU; from the main checkout after merging Tasks 1–5)

**Files:**
- Create: `docs/superpowers/plans/2026-10-09-residual-warm-start-phase1a-RESULTS.md`
- Create (git-ignored): `RESULTS_ROOT/runs/{residual,standard,standard_xhat}_d4_b64/`, eval JSONs, `RESULTS_ROOT/eval/residual/d4/gate_1a.json`, `RESULTS_ROOT/runs/pilot_events.log`

This task spans ≈ 16–20 h of wall-clock (B ‖ A ≈ 7 h each, then A′ ≈ 7 h, with the final full-grid evaluations in the gaps). It is procedural: the driver does the work; the executor checks on it, records what it observed, and writes the note.

- [ ] **Step 1: Start the driver**

Run (from `/data/chaoyi_he/HSI/Diffu`): `nvidia-smi` (both GPUs idle), then `nohup python scripts/pilot/pilot_driver.py --loop 600 >> results/residual_warmstart/runs/pilot_driver.log 2>&1 &`.
Expected: `pilot_events.log` shows `START ('launch', 'residual', 0)` and `START ('launch', 'standard', 1)`; after ~10 min `python scripts/pilot/status.py` shows epoch 1–2 for both with `data` ≤ 10 % and ≈ 256 s/epoch. If `data` exceeds 10 %, stop the driver and both runs (`kill $(cat <run>/pid)`), edit `launch_run.sh` to the worker count Task 3 found compute-bound, delete the two run directories, and restart the driver; record this in the note.

- [ ] **Step 2: Check in every few hours**

`python scripts/pilot/status.py` and `tail results/residual_warmstart/runs/pilot_events.log`. Losses must be finite and falling (compare with the epoch-1 loss in `train_log.jsonl` — spec §11's "B copies x̂₀" check); `skipped` stays 0. A crashed run (alive=False, no DONE) is relaunched by the driver with `--resume auto` on its next pass — note every relaunch. The driver runs the quick evaluations at epochs 25 and 50 by itself; record the quick-grid `ddim10` / `warm100_ddim` RMSE and the y-swap result of each milestone in the note's trajectory table as they appear. No run is stopped or changed on the basis of a milestone evaluation.

- [ ] **Step 3: Let the driver finish**

The driver launches A′ when A is DONE, runs the three full-grid evaluations on idle GPUs, runs the gate, and exits when `gate_1a.json` exists (`pilot_events.log` ends with `END ('gate', …) rc=0`). If the driver itself died (no new events for > 30 min while a run is alive), restart it with the same command — it resumes from the observed state.

- [ ] **Step 4: Write the results note**

`docs/superpowers/plans/2026-10-09-residual-warm-start-phase1a-RESULTS.md`:

```markdown
# Phase 1a results — <date>

## Runs (d = 4, base 64, batch 8, 1 GPU each, 100 epochs or 24 h; no EMA)
| run | GPU | epochs done | wall h | s/epoch (data %) | loss epoch 1 → final | val loss (800 non-gate files) final | peak GB | skipped steps | relaunches |
|---|---|---|---|---|---|---|---|---|---|
| B (residual) | | | | | | | | | |
| A (standard) | | | | | | | | | |
| A′ (standard_xhat) | | | | | | | | | |

## Milestone trajectory (quick grid, 200 cubes × 2 seeds, RMSE %)
| run | epoch | ddim1 | ddim5 | ddim10 | ddim20 | warm100_ddim | y-swap |
|---|---|---|---|---|---|---|---|
| B / A / A′ | 25, 50, final | | | | | | |

## Final full grid (RMSE % ± SE on the 200-cube subset; baseline 2.3512 ± 0.0529; DDPM-1000 on 50 cubes, baseline 2.4280)
| row | NFE | B | A | A′ | B − baseline (paired) | sec/cube B |
|---|---|---|---|---|---|---|
| ddim1 … warm400_ddpm, ddpm1000 | | | | | | |

Per split (B at its selected setting and at ddim10): P022 / P023 / P024 / unseen with SE and paired difference.
y-swap (B): true / swap_full / swap_sensor_only / zero at t = 100 and 300; A and A′ likewise.
Speed–quality: ddim10 − ddpm1000 on the 50-cube intersection per arm (from gate_1a.json).

## Gate 1a
<verbatim gate_1a.py output>. Verdict: PASS | PASS_NOT_OVER_APRIME | FAIL | INVALID. Selected settings are marked as selected from the grid (spec §7); ddim10 is the pre-registered setting. Note: each arm is one training seed; a decisive margin under ≈ 3 SE is within plausible training-seed variance and should be confirmed with a second seed before 1b.

## Observations
- §11 risk checks: B copies x̂₀? (paired gain at every setting); network ignores the sensor? (y-swap, incl. sensor-only for B); seen-vs-unseen gain; few-step vs DDPM-1000.
- Deviations from the plan (worker count, cap, restarts, resumes) with timestamps from pilot_events.log.
- Pilot B at its selected setting vs the 1b threshold 1.9985 %.

## Decision for the user
- PASS → propose Phase 1b: B at base 256, batch 4/GPU, FSDP, 200 epochs (timing from timing_phase1.json: <sec/epoch> → <h>); `--max_hours` set above that budget.
- PASS_NOT_OVER_APRIME → B and A′ are equivalent at this size: choose which to scale (the residual target's advantage, if any, is not visible at base 64) or scale both.
- FAIL → the diagnosis the spec asks for (σ_d scaling, loss trajectory, direct x₀-prediction of the residual) and a proposal.
- INVALID → the condition path is not used; proposal before any full-size run.
```

Fill every cell from `train_log.jsonl`, `pilot_events.log`, the eval JSONs and `gate_1a.json`; no blanks.

- [ ] **Step 5: Commit and push; present to the user**

```bash
git add docs/superpowers/plans/2026-10-09-residual-warm-start-phase1a-RESULTS.md
git commit -m "docs(warmstart): Phase 1a pilot results and gate

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
git push git@github.com:Chaoyi-He1/Diffu_HSI.git <branch>
```

Then present the note: this is the second decision point in the spec (go to full size or stop and diagnose).
