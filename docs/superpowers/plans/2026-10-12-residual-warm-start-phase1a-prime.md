# Residual Warm-Start — Phase 1a′ (Diagnostics + One-Step Residual Regressor) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer what the Phase 1a pilot left open. Step 0: is B's sampling chain centred on its one-step estimate or on r = 0, and is its spread calibrated? Step 1: does a direct one-step residual regressor x̂ = x̂₀ + σ_d·f(x̂₀, y_d), with B's architecture and budget, beat the baseline and B's one-step read-out? Both are decided under a pre-registered, power-adequate validity rule.

**Architecture:** A new training mode `regress` reuses B's U-Net, wrapper, dataset, checkpoints and training loop. Only the objective changes: one deterministic call at a constant t = 999 on a zero x_t placeholder, with an L1 or L2 loss on the residual. The harness gains a one-row `direct` grid and a regressor y-swap. `scripts/diagnose_b.py` samples B's chain K times per cube with the harness's own seeding and estimates the chain's centre and spread. The one-step centre is the noise-free centre of B's t = 999 output, estimated from the same K draws. `scripts/gate_1a_prime.py` compares each regressor with the baseline and with B's ddim1 on the same 200 cubes, and applies the proposed one-step y-swap rule. It also reports, without binding, a comparison with the mean of B's K = 8 one-step draws (8 NFE). The GPU work runs through the existing `launch_run.sh` (now with `RUN_NAME`); no new driver is built.

**Tech Stack:** Python 3.10 (`/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python`), torch 2.8.0+cu126, numpy, pytest; 2 × RTX A4500 (20 GB).

**Spec:** `docs/superpowers/specs/2026-10-08-residual-warm-start-design.md`; §11's "direct x₀-prediction of the residual" is the fallback this plan builds. The plan argues from the Phase 1a results `docs/superpowers/plans/2026-10-09-residual-warm-start-phase1a-RESULTS.md` (Observations 1–2, Decision steps 0, 1 and 3). The binding rulings are in `docs/superpowers/plans/2026-10-09-residual-warm-start-phase1a-LEDGER.md`.

## Deviations from the spec (for the user to approve with this plan)

1. **Validity rule (spec §7/§8/§11 change).** For this phase the binding check is the *proposed one-step y-swap rule* from the Phase 1a note. It runs at the model's single call (t = 999), on 32 stratified cubes, with per-cube values. Two conditions:
   - (a) swap_full − true > 2 paired SE;
   - (b) swap_full − baseline + 2 SE ≥ 0, i.e. under a swapped condition the gain over the baseline vanishes.

   The spec's 0.9 ratio rule is reported but is not binding for the verdicts. B is re-validated under the same rule from its t = 999 y-swap.

   Reason: to pass, the 0.9 rule needs a one-step gain of ≈ 23 % at t = 100 and ≈ 8 % at t = 300. At t = 999 it needs ≈ 5.4 % (calibrated) to ≈ 8 % (B-like shrinkage). B's 4.6 % (7.1 % on the 32 y-swap cubes) cannot pass, even though a swap removes all of its gain (Phase 1a Observation 2).

   Scope: the deviation changes sub-target verdicts and B's verdict, not the go decision. `go_1b` also requires the spec's 0.9 rule at t = 999. A regressor at the 15 % target is predicted at a ratio of 0.75–0.82, so that requirement costs nothing at the target.
2. **Both L1 and L2 regressors.** Spec §4.3 fixes L1, but the metric is RMSE and L1 targets the median. Both are trained side by side, one per GPU, so the comparison costs no extra wall-clock.
3. **Regressor architecture = B's.**
   - Input: [0, x̂₀] (a zero placeholder in B's 64 x_t channels), context y_d, constant t = `REGRESS_T` = 999 (B's one-step call), output r̂, x̂ = clamp(x̂₀ + σ₄·r̂).
   - Trained from scratch with B's hyper-parameters and budget: base 64, batch 8, 100 epochs, lr 1e-4 cosine, seed 42.
   - The regressor and B load into each other's state dicts. Only `meta` (mode `regress`, prediction_type `direct`) tells them apart, and `check_meta` / `check_resume_meta` enforce that.
4. **Diagnostics budget.**
   - K = 8 draws of DDIM-10 and of B's t = 999 one-step output on the 200 cubes.
   - K = 4 draws of DDPM-1000 on the 50 DDPM cubes.
   - Cost: ≈ 55 min of wall-clock over both GPUs. The Phase 1a note's "minutes of GPU" was wrong for DDPM-1000 and is corrected in Task 8.
   - K = 4 already separates the two centre hypotheses by ≥ 7.5 noise SE (groundwork computation).
5. **Second training seed only when it can change the decision.**
   - Seed 43 runs only if the best regressor (the lowest-RMSE `PASS`) also meets the 1.9985 % target. Without the target, seed 43 cannot make `go_1b` true.
   - It then runs for both losses side by side, one per GPU: ≈ 8.8 h, against ≈ 7.2 h for one run alone. This also replicates the L1/L2 comparison and the choice of the better loss.
   - `go_1b` is true when a loss's seed pair qualifies: both seeds `PASS` (hence valid), meet the target and pass the spec's 0.9 rule at t = 999.
6. **No driver.** The pilot driver is hard-coded to the three Phase 1a arms. Runs go through `launch_run.sh`, which gains a `RUN_NAME` override, and a watcher. Diagnostics outputs live under `RESULTS_ROOT/diagnostics/`, which `scripts/yswap_one_step.py` is now allowed to write to.

## Global Constraints

- Everything Phase 1a fixed still binds:
  - The loader is unchanged: `HFDResidualData`, d = 4, 9000 training files, 800 monitoring validation files, 200 gate cubes.
  - `RESULTS_ROOT = /data/chaoyi_he/HSI/Diffu/results/residual_warmstart` (absolute; env `RESIDUAL_WARMSTART_RESULTS`).
  - Prior sha256 `1ad9108f17bee02eceadf63e5cd59671fe3475a9b015aa8c7c9acfd5adaba2b0`, σ₄ = 0.051661.
  - Checkpoints are the `ResidualWrapper` state dict plus `meta`.
  - Distributed decisions are collective; resume is exact.
  - No peeking: the gate uses final checkpoints, and the monitoring loss never sees the gate cubes.
- **The Phase 1a records are read-only inputs and are never overwritten:** B's checkpoint, the three epoch-100 evaluations, `gate_1a__residual_d4_b64.json` and its `__postreview` copy.
- **Comparators on the 200-cube subset:**
  - baseline 2.3512 ± 0.0529 %;
  - B's ddim1 2.2426 ± 0.0507 % (paired −0.1086 ± 0.0060, clustered ± 0.0068);
  - target (the spec §8 1b bar) 0.85 × 2.3512 = **1.9985 %**.
- **Regressor hyper-parameters** (identical to B except the objective):
  - `--mode regress --prediction_type direct --loss_type {l1,l2}`, base 64, batch 8, one GPU, bf16, AdamW lr 1e-4 → 0.033·lr cosine, weight decay 0, grad clip 1.0, seed 42.
  - 100 epochs (24 h cap via `launch_run.sh`), 8 workers, milestones 25/50/100. `--snr_gamma` is ignored by `regress`.
- **Tests:** `cd <worktree> && CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests -q`. All 207 existing tests keep passing. The only edit to an existing test is the `MODES` tuple in `tests/test_wrapper.py` (Task 1).
- **GPU hygiene:**
  - `nvidia-smi` before every GPU command; never start on a GPU with < 3 GB free or another compute process.
  - Every training and sampling process runs with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`. `launch_run.sh` exports these, and the DataLoader `worker_init_fn` limits the workers.
- **Output locations:**
  - New evaluations go to `RESULTS_ROOT/eval/regress/d4/` (new file names only).
  - Diagnostics go to `RESULTS_ROOT/diagnostics/`.
  - Gate records go next to the first `--reg` evaluation.
- **Commits:** one per task (code tasks), explicit paths, never `__pycache__`. The message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP`.
- **Branch:** work continues on `worktree-phase1-plan` (worktree `/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan`); the merge decision stays with the user.

## Review Focus

1. **A regressor checkpoint evaluated as B, or B's as a regressor.** The architectures are identical and a strict load cannot tell them apart. Expected: `check_meta` refuses on `mode` before any sampling. Tests: `test_regress_checkpoint_meta_is_refused_as_residual` (Task 1) and the wrong-mode refusal in `smoke_regress.sh` (Task 5).
2. **Mode/prediction_type/grid mismatches** (regress with `v`, residual with `direct`, regress on the sampler grid) silently producing a wrong run or evaluation. Expected: `ValueError` before anything is loaded. Tests: `test_make_trainer_ties_mode_regress_to_prediction_type_direct` (Task 1) and `test_checkpoint_eval_refuses_mode_grid_and_prediction_type_mismatches` (Task 2).
3. **A regressor y-swap composed with the swapped x̂₀, or evaluated at t = 100/300.** Expected: every variant is composed with the TRUE x̂₀, and the single row is at `REGRESS_T`. Test: `test_yswap_regress_composes_every_variant_with_the_true_x0_hat` (Task 2).
4. **Diagnostic draws that do not reproduce the harness's seeding** (draw k, batch bi → seed k·1000 + bi, batch 8, dataset order). Expected: draws 0 and 1 of a CUDA-sampled shard equal B's stored per-cube values; a CPU shard draws independent noise, and merge warns. Tests: `test_accumulate_uses_the_harness_seeding_and_sums_per_cube` and `test_consistency_reads_the_stored_seed_row_by_file` (Task 3), and the Task 6 consistency check (≤ 1e-3).
5. **A gate fed non-comparable inputs:** different cubes or prior, regressors stopped at different epochs, a non-final checkpoint. Expected: refusal naming the reason. Tests: `test_refusals` and `test_done_is_required` (Task 4).

---

### Task 1: Mode `regress` — shared architecture, objective, guards

**Files:**
- Modify: `model/residual_wrapper.py` (MODES, `REGRESS_T`, `build_residual_model`, `target_for`, new `regress_forward`)
- Create: `model/regression_trainer.py`
- Modify: `main_2d_residual_fsdp.py` (import, `--prediction_type` choices, `check_mode_args`, `make_trainer`, first line of `build_everything`, `check_resume_meta`, module docstring)
- Modify: `tests/test_wrapper.py` (the `MODES` tuple assertion, currently `('residual', 'standard', 'standard_xhat')`)
- Test: `tests/test_regress_mode.py` (new)

**Interfaces:**
- Consumes:
  - `ResidualWrapper(net, use_x0_hat)`, `U2NetHyperspectral`;
  - training-script functions `get_parser`, `make_loader(ds, args, batch_size, shuffle, rank, world, device)`, `make_optimizer`, `make_scheduler`, `epoch_seed`, `train_one_epoch(model, trainer, loader, optimizer, args, device, dist_ctx, mode, sigma_d)`, `validate(model, trainer, loader, args, device, dist_ctx, mode, sigma_d)`, `checkpoint_meta`, `save_checkpoint`, `load_checkpoint`, `check_resume_meta(ck, args, sigma_d, prior_sha256, path=..., warn=True)`, `NO_DIST`;
  - harness `check_meta(ck, args, sigma_d, prior_sha256)`.
- Produces:
  - `MODES = ('residual', 'standard', 'standard_xhat', 'regress')`; `REGRESS_T = 999`;
  - `regress_forward(model, cond) -> r̂`, which calls `model(zeros_like(x0_hat), cond, full(B, REGRESS_T))`;
  - `RegressionTrainer(*, device, loss_type='l2', n_timesteps=1000)`, with `.prediction_type == 'direct'`, `.last_nfe == 1` and `.get_loss(model, x_0, cond=None, t=None, *, loss_in_fp32=False) -> (loss, info)`;
  - `main_2d_residual_fsdp.check_mode_args(args)`. It raises `ValueError` mentioning `direct` unless `(mode == 'regress') == (prediction_type == 'direct')`.
  - `check_resume_meta` also refuses a checkpoint whose `meta['args']` records a different `loss_type` or `seed` (the L1 and L2 runs differ only in `--loss_type`, and Task 7 relaunches dead runs by hand).

- [ ] **Step 1: Write the failing tests**

`tests/test_regress_mode.py`:

```python
"""Mode 'regress' (Phase 1a-prime): the one-step residual regressor shares B's architecture, its target and its
training loop; only the objective differs (one deterministic call at t = REGRESS_T on a zero x_t placeholder)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import main_2d_residual_fsdp as tr
from model.regression_trainer import RegressionTrainer
from model.residual_wrapper import MODES, REGRESS_T, ResidualWrapper, build_residual_model, regress_forward, target_for

CPU = torch.device('cpu')


class LearnableResidualSet(torch.utils.data.Dataset):
    """(x64, y_d, x0_hat) HWC float32 on [-1, 1] whose residual r = (x - x0_hat) / sigma_d = 0.5 * x0_hat is a known
    target (training-loop smoke data)."""
    sigma_d = 0.05
    prior = {'s': 3e-7}

    def __init__(self, n=8, H=16, d=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.xh = torch.rand(n, H, H, 64, generator=g) * 1.6 - 0.8
        self.x = (self.xh + self.sigma_d * 0.5 * self.xh).clamp(-1, 1)
        self.y = torch.rand(n, H // d, H // d, 30, generator=g)
        self.img_list = [f'MatFlower60/Train/P0{22 + i % 3}/f{i}_1.mat' for i in range(n)]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        return self.x[i].numpy(), self.y[i].numpy(), self.xh[i].numpy()


def regress_args(tmp_path, **kw):
    a = tr.get_parser().parse_args(['--mode', 'regress', '--prediction_type', 'direct', '--loss_type', 'l2',
                                    '--device', 'cpu', '--base_channels', '8', '--batch_size', '2', '--num_workers', '0',
                                    '--num_epochs', '3', '--save_dir', str(tmp_path / 'run'), '--no_bf16'])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def parts_for(args, ds=None):
    ds = ds or LearnableResidualSet()
    loader = tr.make_loader(ds, args, batch_size=args.batch_size, shuffle=True, rank=0, world=1, device=CPU)
    model = build_residual_model('regress', args.base_channels)
    trainer = tr.make_trainer(args, CPU)
    opt = tr.make_optimizer(model, args)
    sched = tr.make_scheduler(opt, args)
    return ds, loader, model, trainer, opt, sched


def epoch(parts, args, e):
    ds, loader, model, trainer, opt, sched = parts
    tr.epoch_seed(args, e, 0)
    loader.sampler.set_epoch(e)
    rec = tr.train_one_epoch(model, trainer, loader, opt, args, CPU, tr.NO_DIST, 'regress', ds.sigma_d)
    sched.step()
    return rec


def test_regress_mode_shares_bs_architecture_and_target():
    assert MODES == ('residual', 'standard', 'standard_xhat', 'regress') and REGRESS_T == 999
    m = build_residual_model('regress', base_channels=8)
    assert isinstance(m, ResidualWrapper) and m.use_x0_hat and m.net.input_proj.in_channels == 128
    m.load_state_dict(build_residual_model('residual', base_channels=8).state_dict(), strict=True)   # only meta tells them apart
    x, xh = torch.rand(2, 4, 8, 8) * 2 - 1, torch.rand(2, 4, 8, 8) * 2 - 1
    assert torch.allclose(target_for('regress', x, xh, 0.05), (x - xh) / 0.05)


def test_regress_forward_uses_a_zero_placeholder_and_the_constant_t():
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, x_t, cond, t):
            seen.update(x_t=x_t.clone(), t=t.clone(), keys=sorted(cond))
            return torch.ones_like(x_t)
    xh = torch.rand(3, 64, 16, 16)
    out = regress_forward(Spy(), {'sensor': torch.rand(3, 30, 4, 4), 'x0_hat': xh})
    assert out.shape == xh.shape
    assert torch.equal(seen['x_t'], torch.zeros_like(xh)) and seen['t'].dtype == torch.long
    assert seen['t'].tolist() == [REGRESS_T] * 3 and seen['keys'] == ['sensor', 'x0_hat']


@pytest.mark.parametrize('loss_type', ['l1', 'l2'])
def test_regression_trainer_loss_is_the_plain_per_sample_mean(loss_type):
    pred = torch.randn(2, 64, 16, 16)

    class Const(torch.nn.Module):
        def forward(self, x_t, cond, t):
            return pred.clone()
    r = torch.randn(2, 64, 16, 16)
    cond = {'sensor': torch.rand(2, 30, 4, 4), 'x0_hat': torch.rand(2, 64, 16, 16)}
    trainer = RegressionTrainer(device=CPU, loss_type=loss_type)
    err = (pred - r).abs() if loss_type == 'l1' else (pred - r) ** 2
    torch.manual_seed(1)
    a, info = trainer.get_loss(Const(), r, cond, loss_in_fp32=True)
    torch.manual_seed(2)
    b, _ = trainer.get_loss(Const(), r, cond, loss_in_fp32=True)
    assert torch.allclose(a, err.mean()) and float(a) == float(b)                  # deterministic: no noise, no t draw
    assert trainer.prediction_type == 'direct' and trainer.last_nfe == 1 and info['t'] == REGRESS_T
    with pytest.raises(ValueError):
        RegressionTrainer(device=CPU, loss_type='huber')


def test_make_trainer_ties_mode_regress_to_prediction_type_direct(tmp_path):
    t = tr.make_trainer(regress_args(tmp_path), CPU)
    assert isinstance(t, RegressionTrainer) and t.loss_type == 'l2'
    for bad in (dict(prediction_type='v'), dict(mode='residual')):                 # regress + v, residual + direct
        with pytest.raises(ValueError, match='direct'):
            tr.make_trainer(regress_args(tmp_path, **bad), CPU)
    with pytest.raises(ValueError, match='direct'):
        tr.check_mode_args(regress_args(tmp_path, prediction_type='x0'))


def test_regress_training_runs_and_its_loss_falls(tmp_path):
    """Smoke only: at base 8 the 8-channel input_proj cannot carry the 64 independent channels of r = 0.5 x0_hat, so
    the loss falls toward the x0_hat-blind floor 0.25 E[x0_hat^2], not below it. The training call itself is pinned
    by the spy test below; use of x0_hat by the y-swap Oracle test (Task 2)."""
    args = regress_args(tmp_path, lr=3e-3, num_epochs=8)
    torch.manual_seed(0)
    parts = parts_for(args)
    losses = [epoch(parts, args, e)['loss'] for e in range(8)]
    assert all(np.isfinite(losses)) and min(losses[-3:]) < 0.8 * losses[0], losses


def test_regress_training_step_calls_the_net_on_zero_x_t_at_regress_t_against_the_residual(tmp_path):
    seen = []

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x_t, cond, t):
            seen.append((x_t.detach().clone(), t.clone()))
            return torch.zeros_like(x_t) + self.p
    args = regress_args(tmp_path, lr=0.0)
    ds = LearnableResidualSet()
    loader = tr.make_loader(ds, args, batch_size=2, shuffle=True, rank=0, world=1, device=CPU)
    model = Spy()
    tr.epoch_seed(args, 0, 0)
    loader.sampler.set_epoch(0)
    rec = tr.train_one_epoch(model, tr.make_trainer(args, CPU), loader, tr.make_optimizer(model, args), args, CPU,
                             tr.NO_DIST, 'regress', ds.sigma_d)
    assert len(seen) == rec['steps'] == 4
    assert all(float(x.abs().max()) == 0.0 and t.tolist() == [REGRESS_T] * 2 for x, t in seen)
    assert rec['loss'] == pytest.approx(0.25 * float((ds.xh ** 2).mean()), rel=1e-4)   # zero predictor vs r = 0.5 x0_hat


def test_regress_validate_is_deterministic(tmp_path):
    args = regress_args(tmp_path)
    ds, _, model, trainer, _, _ = parts_for(args)
    vl = tr.make_loader(ds, args, batch_size=2, shuffle=False, rank=0, world=1, device=CPU)
    a = tr.validate(model, trainer, vl, args, CPU, tr.NO_DIST, 'regress', ds.sigma_d)
    torch.manual_seed(5)
    torch.rand(4)
    b = tr.validate(model, trainer, vl, args, CPU, tr.NO_DIST, 'regress', ds.sigma_d)
    assert a['val_loss'] == b['val_loss'] and a['val_n'] == 8


def test_regress_checkpoint_meta_is_refused_as_residual(tmp_path):
    from scripts.eval_warmstart import check_meta
    args = regress_args(tmp_path)
    parts = parts_for(args)
    ds, _, model, _, opt, sched = parts
    epoch(parts, args, 0)
    sha = 'ab' * 32
    meta = tr.checkpoint_meta(args, ds.sigma_d, ds.prior['s'], sha)
    assert meta['mode'] == 'regress' and meta['prediction_type'] == 'direct' and meta['args']['loss_type'] == 'l2'
    path = str(tmp_path / 'run' / 'checkpoint_epoch_1.pth')
    tr.save_checkpoint(model, opt, sched, 0, 0.5, meta, path, 0.01, tr.NO_DIST)
    ck = tr.load_checkpoint(path)
    assert tr.check_resume_meta(ck, args, ds.sigma_d, sha, warn=False) is True
    with pytest.raises(ValueError, match='mode'):
        tr.check_resume_meta(ck, regress_args(tmp_path, mode='residual', prediction_type='v'), ds.sigma_d, sha, warn=False)
    with pytest.raises(ValueError, match='loss_type'):          # the L1 and L2 runs differ only in --loss_type
        tr.check_resume_meta(ck, regress_args(tmp_path, loss_type='l1'), ds.sigma_d, sha, warn=False)
    with pytest.raises(ValueError, match='seed'):
        tr.check_resume_meta(ck, regress_args(tmp_path, seed=43), ds.sigma_d, sha, warn=False)
    eval_args = SimpleNamespace(ckpt=path, mode='residual', d=4, base_channels=8, prediction_type='v')
    with pytest.raises(ValueError, match="mode: checkpoint 'regress'"):
        check_meta(ck, eval_args, ds.sigma_d, sha)
    assert check_meta(ck, SimpleNamespace(**dict(vars(eval_args), mode='regress', prediction_type='direct')),
                      ds.sigma_d, sha) is True
```

Edit `tests/test_wrapper.py`: the assertion `MODES == ('residual', 'standard', 'standard_xhat')` becomes `MODES == ('residual', 'standard', 'standard_xhat', 'regress')`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_regress_mode.py tests/test_wrapper.py -q`
Expected: collection error `ImportError` (`REGRESS_T`, `regress_forward`, `model.regression_trainer`).

- [ ] **Step 3: Implement**

`model/residual_wrapper.py`: replace the `MODES` block, `build_residual_model`'s `use_x0_hat` line and `target_for`, and append `regress_forward`:

```python
MODES = ('residual', 'standard', 'standard_xhat', 'regress')
# residual      (B):  input [x_t, x0_hat], target (x - x0_hat) / sigma_d
# standard      (A):  input  x_t,          target  x
# standard_xhat (A'): input [x_t, x0_hat], target  x   -- B's inputs with A's target: controls for the spatially
#                     aligned x0_hat channels that the position-blind cross-attention context cannot provide
# regress       (R):  input [0, x0_hat] at t = REGRESS_T, target (x - x0_hat) / sigma_d predicted in ONE deterministic
#                     call (no noise, no sampler): B's architecture in B's one-step regime, trained as a regressor

REGRESS_T = 999     # the regressor's constant time input: B's one-step (ddim1) call is at t = 999, where x_t is pure noise
```

In `build_residual_model`: `use_x0_hat = mode in ('residual', 'standard_xhat', 'regress')`.

```python
def target_for(mode, x, x0_hat, sigma_d):
    """The training target per mode (all on the loader's [-1, 1] scale)."""
    if mode in ('residual', 'regress'):
        return (x - x0_hat) / sigma_d
    return x


def regress_forward(model, cond):
    """The regressor's single call: a zero x_t placeholder (the 64 channels B reserves for x_t) at t = REGRESS_T.
    Training, reconstruction and the y-swap all call this, so the placeholder and t cannot drift apart."""
    xh = cond['x0_hat']
    t = torch.full((xh.shape[0],), REGRESS_T, dtype=torch.long, device=xh.device)
    return model(torch.zeros_like(xh), cond, t)
```

`model/regression_trainer.py`:

```python
"""Objective of the one-step residual regressor (mode 'regress'), duck-typed to DiffusionTrainer.get_loss so the
training script's loops (train_one_epoch, validate, dry_run) run it unchanged."""
from model.residual_wrapper import REGRESS_T, regress_forward


class RegressionTrainer:
    prediction_type = 'direct'

    def __init__(self, *, device, loss_type='l2', n_timesteps=1000):
        if loss_type not in ('l1', 'l2'):
            raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type!r}")
        self.device = device
        self.loss_type = loss_type
        self.n_timesteps = int(n_timesteps)       # provenance only: the regressor has no noise schedule
        self.last_nfe = 1

    def get_loss(self, model, x_0, cond=None, t=None, *, loss_in_fp32=False):
        """x_0 is the regression target r = (x - x0_hat) / sigma_d; `t` is ignored (the call is at REGRESS_T).
        Returns (mean over samples of the per-sample mean |r_hat - r| or (r_hat - r)^2, info)."""
        x_0 = x_0.to(self.device)
        pred = regress_forward(model, cond)
        p, r = (pred.float(), x_0.float()) if loss_in_fp32 else (pred, x_0)
        err = (p - r).abs() if self.loss_type == 'l1' else (p - r) ** 2
        loss = err.reshape(err.shape[0], -1).mean(dim=1).mean()
        return loss, dict(loss=loss.detach(), pred=pred.detach(), t=REGRESS_T)
```

`main_2d_residual_fsdp.py`:
- Import: `from model.regression_trainer import RegressionTrainer  # noqa: E402`.
- The `--prediction_type` argument becomes `ap.add_argument('--prediction_type', choices=['eps', 'x0', 'v', 'direct'], default='v', help="diffusion parameterisation; 'direct' is required by, and only allowed with, --mode regress")`.
- The `--snr_gamma` help gains `(ignored by --mode regress)`.
- Replace `make_trainer`:

```python
def check_mode_args(args):
    """--mode regress and --prediction_type direct go together (the meta records both; check_meta compares both)."""
    if (args.mode == 'regress') != (args.prediction_type == 'direct'):
        raise ValueError(f"--mode regress needs --prediction_type direct, and 'direct' needs --mode regress "
                         f"(got mode={args.mode!r}, prediction_type={args.prediction_type!r})")


def make_trainer(args, device):
    check_mode_args(args)
    if args.mode == 'regress':
        return RegressionTrainer(device=device, loss_type=args.loss_type, n_timesteps=N_TIMESTEPS)
    return DiffusionTrainer(device=device, n_timesteps=N_TIMESTEPS, loss_type=args.loss_type,
                            prediction_type=args.prediction_type, snr_gamma=args.snr_gamma if args.snr_gamma > 0 else None)
```

- First line of `build_everything`: `check_mode_args(args)`, so a mismatch fails before the datasets are built.
- In `check_resume_meta`, directly after the `bad = [...]` list over `RESUME_META_KEYS`:

```python
    margs = meta.get('args') or {}
    for k in ('loss_type', 'seed'):      # meta['args'] is vars(args) at save time; the L1/L2 runs differ only in loss_type
        if k in margs and margs[k] != getattr(args, k, None):
            bad.append(f'{k}: checkpoint {margs[k]!r} vs args {getattr(args, k, None)!r}')
```

  In its docstring, "Compares RESUME_META_KEYS exactly" becomes "Compares RESUME_META_KEYS and meta['args'] loss_type and seed exactly" (rewrap the docstring to the file's 120-column limit).
- Module docstring: add the line `Mode regress (R, the one-step residual regressor): --mode regress --prediction_type direct --loss_type l1|l2.`

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_regress_mode.py tests/test_wrapper.py -q`, then the full suite.
Expected: the 9 new tests pass; full suite 216 passed (207 + 9).

- [ ] **Step 5: Commit**

```bash
git add model/regression_trainer.py tests/test_regress_mode.py
git commit -F <msgfile> -- model/residual_wrapper.py model/regression_trainer.py main_2d_residual_fsdp.py tests/test_wrapper.py tests/test_regress_mode.py
```
Message: `feat(warmstart): mode regress — one-step residual regressor with B's architecture (zero x_t placeholder at t = 999), L1/L2 objective, mode/prediction_type guard; resume refuses another loss_type or seed` + blank line + the two trailer lines.

---

### Task 2: Harness — `direct` grid, one-call reconstruction, regressor y-swap, guards, diagnostics output dir

**Files:**
- Modify: `scripts/eval_warmstart.py` (import, docstring, `sampler_grid`, `reconstruct`, `yswap_check`, `checkpoint_eval`, `main`)
- Modify: `scripts/yswap_one_step.py` (`DIAG_DIR`, the `--out` guard, docstring)
- Test: `tests/test_harness_regress.py` (new)

**Interfaces:**
- Consumes: `REGRESS_T`, `regress_forward` (Task 1).
- Produces:
  - `sampler_grid('direct') == [dict(name='direct', method='direct', n_steps=1)]`.
  - `reconstruct(..., mode='regress', cfg=direct)` → clamp(x̂₀ + σ·r̂), with `trainer.last_nfe = 1` and no grad.
  - `yswap_check(..., mode='regress', sigma_d=σ)` → a record at the single t = `REGRESS_T`, with every variant composed with the true x̂₀.
  - `checkpoint_eval` refusals (`ValueError` naming `grid` or `prediction_type`) before anything is loaded.
  - Provenance gains `loss_type`, `train_seed` (from the checkpoint meta args) and `yswap_n`.
  - CLI: `--grid {full,quick,direct}` and `--yswap_n` (default 32 for regress, 8 otherwise).
  - `scripts/yswap_one_step.DIAG_DIR = RESULTS_ROOT/diagnostics`; `--out` may lie there.

- [ ] **Step 1: Write the failing tests**

`tests/test_harness_regress.py`:

```python
"""Harness support for mode 'regress' (Phase 1a-prime Task 2): the one-row 'direct' grid, the one-call reconstruction,
the regressor's y-swap, the mode/grid/prediction_type guards and the y-swap script's diagnostics output directory."""
import argparse
import json
import os

import pytest
import torch

from model.diffusion_trainer import DiffusionTrainer
from model.residual_wrapper import REGRESS_T
from scripts import eval_warmstart as ew

CPU = torch.device('cpu')
DIRECT = dict(name='direct', method='direct', n_steps=1)


class Const(torch.nn.Module):
    """Predicts a constant residual; has a parameter so a missing no_grad shows up as requires_grad."""

    def __init__(self, v):
        super().__init__()
        self.v = v
        self.p = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x_t, cond, t):
        return torch.full_like(cond['x0_hat'], self.v) + 0 * self.p


def test_direct_grid_is_one_call():
    assert ew.sampler_grid('direct') == [DIRECT]
    assert 'direct' not in [c['name'] for c in ew.sampler_grid('full')]
    with pytest.raises(ValueError):
        ew.sampler_grid('fast')


def test_reconstruct_direct_is_x0_hat_plus_sigma_times_one_prediction():
    tr = DiffusionTrainer(device=CPU, prediction_type='direct', snr_gamma=None)
    xh = torch.rand(2, 4, 8, 8) * 1.2 - 0.6
    batch = (torch.zeros(2, 4, 8, 8), torch.zeros(2, 3, 2, 2), xh)
    out = ew.reconstruct(Const(3.0), tr, 'regress', batch, DIRECT, sigma_d=0.1, seed=0)
    assert torch.allclose(out, (xh + 0.3).clamp(-1, 1)) and tr.last_nfe == 1 and not out.requires_grad
    with pytest.raises(ValueError, match='regress'):
        ew.reconstruct(Const(3.0), tr, 'residual', batch, DIRECT, 0.1, 0)
    with pytest.raises(ValueError, match='regress'):
        ew.reconstruct(Const(3.0), tr, 'regress', batch, dict(name='ddim1', method='ddim', n_steps=1), 0.1, 0)


def test_reconstruct_direct_clamps_to_the_cube_range():
    tr = DiffusionTrainer(device=CPU, prediction_type='direct', snr_gamma=None)
    xh = torch.rand(2, 4, 8, 8) * 1.2 - 0.6
    out = ew.reconstruct(Const(30.0), tr, 'regress', (torch.zeros_like(xh), torch.zeros(2, 3, 2, 2), xh), DIRECT,
                         sigma_d=0.1, seed=0)
    assert torch.equal(out, torch.ones_like(xh))


def test_yswap_regress_composes_every_variant_with_the_true_x0_hat():
    tr = DiffusionTrainer(device=CPU, prediction_type='direct', snr_gamma=None)
    torch.manual_seed(0)
    x = torch.rand(4, 4, 8, 8) * 1.6 - 0.8
    batch = (x, torch.rand(4, 3, 2, 2), 0.9 * x)                  # x0_hat = 0.9 x and sigma_d = 0.1, so r = x

    class Oracle(torch.nn.Module):                                 # reads its condition: f(x0_hat) = x0_hat / 0.9
        def forward(self, x_t, cond, t):
            assert t.tolist() == [REGRESS_T] * x_t.shape[0] and float(x_t.abs().max()) == 0.0
            return cond['x0_hat'] / 0.9
    files = [f'MatFlower60/Train/P0{22 + i % 3}/f{i}_1.mat' for i in range(4)]
    res = ew.yswap_check(Oracle(), tr, 'regress', batch, t_values=(100, 300), sigma_d=0.1, files=files)
    assert set(res) == {f't{REGRESS_T}', 'passes', 'per_cube', 'baseline_per_cube', 'files'}     # t forced to REGRESS_T
    cubes = res['per_cube'][str(REGRESS_T)]
    assert max(cubes['true']) < 1e-4 and min(cubes['swap_full']) > 1e-2 and res['passes']
    blind = ew.yswap_check(Const(0.0), tr, 'regress', batch, sigma_d=0.1)
    c = blind['per_cube'][str(REGRESS_T)]
    assert c['true'] == c['swap_full'] == c['swap_sensor_only'] and not blind['passes']
    with pytest.raises(ValueError, match='sigma_d'):
        ew.yswap_check(Oracle(), tr, 'regress', batch)


def test_checkpoint_eval_refuses_mode_grid_and_prediction_type_mismatches(tmp_path):
    base = dict(d=4, ckpt=str(tmp_path / 'missing.pth'), base_channels=8, seeds=[0, 1], batch_size=4, device='cpu',
                data_root=str(tmp_path), prior=str(tmp_path / 'prior.npz'), run_name=None, yswap_n=None)
    for kw, msg in ((dict(mode='regress', grid='full', prediction_type='direct'), 'grid'),
                    (dict(mode='residual', grid='direct', prediction_type='v'), 'grid'),
                    (dict(mode='regress', grid='direct', prediction_type='v'), 'prediction_type'),
                    (dict(mode='residual', grid='full', prediction_type='direct'), 'prediction_type')):
        with pytest.raises(ValueError, match=msg):
            ew.checkpoint_eval(argparse.Namespace(**base, **kw))     # refused before anything is loaded


def test_yswap_one_step_writes_only_under_the_diagnostics_dir(tmp_path, monkeypatch):
    from scripts import yswap_one_step as ys
    root = tmp_path / 'root'
    monkeypatch.setattr(ys, 'RESULTS_ROOT', str(root))
    monkeypatch.setattr(ys, 'DIAG_DIR', str(root / 'diagnostics'))
    missing = str(tmp_path / 'missing.pth')
    with pytest.raises(SystemExit, match='results tree'):
        ys.main(['--checkpoint', missing, '--mode', 'residual', '--out', str(root / 'eval' / 'x.json')])
    with pytest.raises(FileNotFoundError):                       # passes the guard, then fails on the missing checkpoint
        ys.main(['--checkpoint', missing, '--mode', 'residual', '--out', str(root / 'diagnostics' / 'x.json')])


@pytest.mark.dataset
def test_checkpoint_eval_regress_writes_a_one_row_direct_evaluation(data_root, tmp_path, monkeypatch):
    from data_loader.linear_estimate import load_prior
    from data_loader.warmstart_paths import PRIOR
    if not os.path.exists(PRIOR):
        pytest.skip('prior missing')
    torch.manual_seed(0)
    sd = ew.build_model('regress', 8, CPU).state_dict()
    p = load_prior(PRIOR)
    meta = dict(mode='regress', d=4, base_channels=8, prediction_type='direct',
                sigma_d=float(p['sigma_d'][list(p['d_values']).index(4)]), prior_sha256=ew.file_sha256(PRIOR),
                args=dict(loss_type='l2', seed=42))
    ckpt = tmp_path / 'tiny.pth'
    torch.save({'model_state_dict': sd, 'meta': meta}, ckpt)
    real = ew.load_subset
    pick = list(range(0, 3)) + list(range(24, 27)) + list(range(176, 178))              # 3 P022, 3 P023, 2 P024
    monkeypatch.setattr(ew, 'load_subset', lambda root, which='files': (
        [real(root, 'files')[i] for i in pick] if which == 'files' else real(root, which)[:4]))
    monkeypatch.setattr(ew, 'subset_class_counts', lambda: {'P022': 3, 'P023': 3, 'P024': 2})
    monkeypatch.setattr(ew, 'EVAL_DIR', str(tmp_path))
    args = argparse.Namespace(d=4, ckpt=str(ckpt), mode='regress', grid='direct', run_name='tiny_run', base_channels=8,
                              prediction_type='direct', seeds=[0, 1], batch_size=4, device='cpu', data_root=data_root,
                              prior=PRIOR, yswap_n=None)
    ew.checkpoint_eval(args)
    out = json.load(open(tmp_path / 'regress' / 'd4' / 'tiny_run__tiny__direct.json'))
    assert [r['name'] for r in out['rows']] == ['direct'] and out['rows'][0]['nfe'] == 1
    assert out['grid'] == 'direct' and out['provenance']['loss_type'] == 'l2' and out['provenance']['train_seed'] == 42
    assert out['provenance']['meta_checked'] is True and out['provenance']['prediction_type'] == 'direct'
    assert out['provenance']['yswap_n'] == 32 and len(out['yswap']['files']) == 8   # regress default; all 8 picked cubes
    ys = out['yswap']
    assert set(ys) >= {f't{REGRESS_T}', 'per_cube', 'baseline_per_cube', 'files'}
    assert len(ys['files']) == len(ys['baseline_per_cube'])
    assert out['rows'][0]['rmse_pct_seed_std'] == 0.0                          # deterministic: both seeds identical
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_harness_regress.py -q`
Expected: failures (`ValueError: grid must be 'full' or 'quick'`, missing guards, the y-swap computed at t = 100/300 with q_sample, the missing `DIAG_DIR`).

- [ ] **Step 3: Implement**

`scripts/eval_warmstart.py`:
- Import line: `from model.residual_wrapper import MODES, REGRESS_T, build_residual_model, regress_forward  # noqa: E402`.
- Docstring usage gains: `python scripts/eval_warmstart.py --d 4 --ckpt <run>/checkpoint_epoch_100.pth --mode regress --prediction_type direct --grid direct` and the sentence `Mode regress (the one-step regressor) is scored on the one-row direct grid; its y-swap is one deterministic call at t = REGRESS_T on 32 cubes (--yswap_n).`

`sampler_grid`, replaced:

```python
def sampler_grid(which='full'):
    if which == 'direct':                        # the one-step regressor: one deterministic call, no sampler
        return [dict(name='direct', method='direct', n_steps=1)]
    g = [dict(name='ddpm1000', method='ddpm', n_steps=1000)]
    g += [dict(name=f'ddim{k}', method='ddim', n_steps=k) for k in (1, 2, 5, 10, 20)]
    g += [dict(name=f'warm{t}_ddim', method='ddim', n_steps=10, t_start=t) for t in (50, 100, 200, 400)]
    g += [dict(name=f'warm{t}_ddpm', method='ddpm', n_steps=1000, t_start=t) for t in (50, 100, 200, 400)]
    if which == 'full':
        return g
    if which == 'quick':                         # milestone checks: every row costs <= 20 NFE, no 50-cube DDPM row
        return [c for c in g if c['name'] in QUICK_GRID]
    raise ValueError(f"grid must be 'full', 'quick' or 'direct', got {which!r}")
```

`reconstruct`, replaced:

```python
def reconstruct(model, trainer, mode, batch, cfg, sigma_d, seed):
    """batch = (x, y_d, x0_hat) tensors. Returns x_hat [B, 64, H, W] clamped to [-1, 1]. Mode regress takes the
    'direct' cfg only (one deterministic call; `seed` unused) and every other mode takes a sampler cfg."""
    x, y_d, x0_hat = [b.to(trainer.device) for b in batch]
    cond = {'sensor': y_d, 'x0_hat': x0_hat}
    if (cfg['method'] == 'direct') != (mode == 'regress'):
        raise ValueError(f"the direct cfg is for mode 'regress' only, and mode 'regress' takes only the direct cfg "
                         f"(mode={mode!r}, method={cfg['method']!r})")
    if mode == 'regress':
        with torch.no_grad():
            r = regress_forward(model, cond)
        trainer.last_nfe = 1
        return (x0_hat + sigma_d * r).clamp(-1, 1)        # clamping the cube = clamping r to the feasible envelope
    torch.manual_seed(seed)
    kw = dict(n_steps=cfg['n_steps'], method=cfg['method'], progress=False)
    if 't_start' in cfg:
        kw['t_start'] = cfg['t_start']
        kw['x_init'] = torch.zeros_like(x0_hat) if mode == 'residual' else x0_hat
    if mode == 'residual':
        # The residual target (x - x0_hat) / sigma_d is unit-RMS and unbounded in [-1, 1]; both x and x0_hat lie in
        # [-1, 1], so |residual| <= 2 / sigma_d is the exact scalar envelope of feasible residuals.
        kw['clamp_x0'] = (-2.0 / sigma_d, 2.0 / sigma_d)
    out = trainer.sample(model, cond, tuple(x0_hat.shape), **kw)
    if mode == 'residual':
        out = x0_hat + sigma_d * out
    return out.clamp(-1, 1)
```

`yswap_check`: four edits, the rest unchanged.
- The sigma_d check becomes `if mode in ('residual', 'regress') and sigma_d is None: raise ValueError('yswap_check in residual and regress modes needs sigma_d (the residual scale of this d)')`.
- Directly after it:
  ```python
  if mode == 'regress':
      t_values = (REGRESS_T,)          # one deterministic call: the noise level does not apply
  ```
- The target line: `target = (x - x0_hat) / sigma_d if mode in ('residual', 'regress') else x`.
- The inner loop body becomes:

```python
        for name, c in conds.items():
            if mode == 'regress':
                with torch.no_grad():
                    pred = regress_forward(model, c)
            else:
                pred = _one_step_x0(model, trainer, mode, target, c, tv)
            x_hat = (x0_hat + sigma_d * pred) if mode in ('residual', 'regress') else pred
            rmse = metrics(x_hat.clamp(-1, 1), x)['rmse_pct']
            row[name], cubes[name] = float(rmse.mean()), rmse.tolist()
```

The docstring gains: `Mode regress: one deterministic call per variant at t = REGRESS_T (t_values is ignored).`

`checkpoint_eval`: these are the first lines, before `DiffusionTrainer`:

```python
    grid = getattr(args, 'grid', 'full')
    if (args.mode == 'regress') != (grid == 'direct'):
        raise ValueError(f"--mode regress needs --grid direct, and --grid direct needs --mode regress "
                         f"(got mode={args.mode!r}, grid={grid!r})")
    if (args.mode == 'regress') != (args.prediction_type == 'direct'):
        raise ValueError(f"--mode regress needs --prediction_type direct, and 'direct' needs --mode regress "
                         f"(got mode={args.mode!r}, prediction_type={args.prediction_type!r})")
    yswap_n = getattr(args, 'yswap_n', None) or (32 if args.mode == 'regress' else 8)
```

The old `grid = getattr(args, 'grid', 'full')` line is removed.
- After `ck = torch.load(...)` add `targs = ((ck.get('meta') or {}).get('args') or {}) if isinstance(ck, dict) else {}`.
- The `provenance` dict gains `loss_type=targs.get('loss_type'), train_seed=targs.get('seed'), yswap_n=yswap_n`.
- `ds_ys, batch = yswap_batch(args.data_root, args.prior, args.d)` becomes `ds_ys, batch = yswap_batch(args.data_root, args.prior, args.d, yswap_n)`.

`main`:
- `ap.add_argument('--grid', choices=['full', 'quick', 'direct'], default='full')`
- `ap.add_argument('--prediction_type', default='v', help="'direct' for --mode regress")`
- `ap.add_argument('--yswap_n', type=int, default=None, help='y-swap cubes (default 32 for --mode regress, 8 otherwise)')`

`scripts/yswap_one_step.py`:
- After the imports: `DIAG_DIR = os.path.join(RESULTS_ROOT, 'diagnostics')   # the one place in the results tree this script may write`.
- The guard in `main` becomes:

```python
    if under(args.out, RESULTS_ROOT) and not under(args.out, DIAG_DIR):
        raise SystemExit(f'--out {args.out} is inside the results tree {RESULTS_ROOT}; this diagnostic writes there only '
                         f'under {DIAG_DIR}')
```

- In the docstring, "It refuses an --out inside the results tree (RESULTS_ROOT)" becomes "It refuses an --out inside the results tree (RESULTS_ROOT) other than RESULTS_ROOT/diagnostics". Add: "Mode regress needs --prediction_type direct; its y-swap is the single call at t = 999 (--t is ignored)."

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_harness_regress.py tests/test_eval_harness.py -q`, then the full suite.
Expected: the 7 new tests pass; the existing harness tests still pass; full suite 223 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_harness_regress.py
git commit -F <msgfile> -- scripts/eval_warmstart.py scripts/yswap_one_step.py tests/test_harness_regress.py
```
Message: `feat(warmstart): harness scores mode regress — one-row direct grid, one-call reconstruction, regressor y-swap at t = 999 on 32 cubes, mode/grid/prediction_type guards; y-swap script may write under diagnostics/` + trailers.

---

### Task 3: Step-0 diagnostics script `scripts/diagnose_b.py`

**Files:**
- Create: `scripts/diagnose_b.py`
- Test: `tests/test_diagnose_b.py`

**Interfaces:**
- Consumes: `reconstruct`, `sampler_grid`, `metrics`, `dataset_for`, `load_subset`, `make_eval_loader`, `build_model`, `check_meta`, `_mean_se` (harness); `cluster_se`, `source_image` (`scripts/paired_stats.py`); `DiffusionTrainer`.
- Produces:
  - `accumulate(batches, recon, draws, n_cubes) -> (sum_x, sum_x2, err)`;
  - `row_stats(sum_x, sum_x2, err, x) -> dict(K, o, ebar, spread, c2, kmean_rmse, single_rmse)`;
  - `centre_lambda(b, c2, c2_one_step) -> float`;
  - `jackknife_se(stat, files) -> float | None`;
  - `merge_shards(shards)`, `save_shard(path, rows, meta)`, `load_shard(path)`, `check_out(path)`;
  - CLI `sample` / `merge`. The output is `RESULTS_ROOT/diagnostics/b_diagnostics.json`. Per row it holds:
    - `lambda_vs_one_step_centre` (rows other than t999);
    - `lambda_vs_ddim1_single_draw` (a labelled, biased reference);
    - `calibration` (`ratio_of_means`, `self_ratio`, `per_cube_slope_on_b2` ± jackknife SE, `reference_slope_c2_one_step_on_b2`, `per_cube_corr`);
    - `kmean_minus_single`, `pred_h0` / `pred_h1`, `mean_c2_one_step`, `consistency_vs_stored` and `per_cube`.
  - merge refuses shards without the t999 row, whose per-cube c2 is the one-step centre energy.

- [ ] **Step 1: Write the failing tests**

`tests/test_diagnose_b.py`:

```python
"""Step-0 diagnostics (Phase 1a-prime Task 3): seeding, sums, the centre/spread estimates, lambda, shard handling,
the stored-evaluation consistency check and the merge statistics."""
import json
import os

import numpy as np
import pytest
import torch

from scripts import diagnose_b as db


def test_accumulate_uses_the_harness_seeding_and_sums_per_cube():
    x = torch.rand(5, 3, 4, 4) * 2 - 1
    batches = [(x[:2], None, None), (x[2:4], None, None), (x[4:], None, None)]
    seen = []

    def recon(batch, seed):
        seen.append(seed)
        g = torch.Generator().manual_seed(seed)
        return (batch[0] + 0.1 * torch.randn(batch[0].shape, generator=g)).clamp(-1, 1)
    sum_x, sum_x2, err = db.accumulate(batches, recon, [0, 3], 5)
    assert seen == [0, 1, 2, 3000, 3001, 3002]                   # draw k, batch bi -> seed k * 1000 + bi
    xs = [torch.cat([recon(b, k * 1000 + bi) for bi, b in enumerate(batches)]).double().numpy() for k in (0, 3)]
    assert np.allclose(sum_x, xs[0] + xs[1])
    assert np.allclose(sum_x2, sum((v ** 2).reshape(5, -1).mean(1) for v in xs))
    assert np.allclose(err[1], db.metrics(torch.from_numpy(xs[1]), x.double())['rmse_pct'].numpy())
    with pytest.raises(ValueError, match='expected 6'):
        db.accumulate(batches, recon, [0], 6)


def test_row_stats_recovers_centre_and_spread():
    rng = np.random.default_rng(0)
    N, K, shape = 6, 64, (4, 16, 16)
    x = rng.uniform(-0.5, 0.5, (N,) + shape)
    draws = (x + 0.02)[None] + 0.03 * rng.standard_normal((K, N) + shape)    # centre 0.02 off the truth, spread 0.03
    sum_x = draws.sum(0)
    sum_x2 = (draws ** 2).reshape(K, N, -1).mean(2).sum(0)
    err = 50 * np.sqrt(((draws - x[None]) ** 2).reshape(K, N, -1).mean(2))
    s = db.row_stats(sum_x, sum_x2, err, x)
    assert s['K'] == K
    assert np.allclose(s['c2'], 2500 * 0.02 ** 2, rtol=0.05)
    assert np.allclose(s['spread'], 2500 * 0.03 ** 2, rtol=0.05)
    assert np.allclose(s['o'], 2500 * (0.02 ** 2 + 0.03 ** 2), rtol=0.05)


def test_identical_draws_have_zero_spread_and_lambda_endpoints():
    x, d = np.zeros((3, 2, 4, 4)), np.full((3, 2, 4, 4), 0.1)
    s = db.row_stats(4 * d, np.full(3, 4 * 0.01), np.full((4, 3), 50 * 0.1), x)
    assert np.allclose(s['spread'], 0) and np.allclose(s['c2'], 2500 * 0.01) and np.allclose(s['kmean_rmse'], 5.0)
    b, m = np.array([2.0, 3.0]), np.array([1.0, 2.0])
    assert db.centre_lambda(b, b ** 2, m ** 2) == pytest.approx(0.0)                # centred on r = 0
    assert db.centre_lambda(b, m ** 2, m ** 2) == pytest.approx(1.0)                # centred on the one-step centre
    with pytest.raises(ValueError, match='2 draws'):
        db.row_stats(d, np.full(3, 0.01), np.full((1, 3), 5.0), x)


def test_row_stats_spread_is_unbiased_with_two_draws():
    x = np.zeros((1, 1, 1, 2))
    d0, d1 = np.full((1, 1, 1, 2), 0.1), np.full((1, 1, 1, 2), 0.3)
    s = db.row_stats(d0 + d1, (d0 ** 2 + d1 ** 2).reshape(1, -1).mean(1), 50 * np.array([[0.1], [0.3]]), x)
    assert s['spread'][0] == pytest.approx(2500 * 0.02)                   # var({0.1, 0.3}, ddof=1) = 0.02
    assert s['c2'][0] == pytest.approx(2500 * 0.2 ** 2 - 2500 * 0.02 / 2)


def test_jackknife_se_over_source_images():
    files = [f'MatFlower60/Train/P022/{img}_{p}.mat' for img in ('a', 'b', 'c') for p in (1, 2)]
    v = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    # leave-one-image-out means 2.5, 2.0, 1.5 -> jackknife SE = sqrt(2/3 * 0.5)
    assert db.jackknife_se(lambda mask: v[mask].mean(), files) == pytest.approx(np.sqrt(2 / 3 * 0.5))
    assert db.jackknife_se(lambda mask: v[mask].mean(), files[:2]) is None          # one image: no SE


def test_merge_requires_disjoint_draws_and_equal_cubes():
    def shard(draws, files=('a', 'b')):
        K = len(draws)
        return {'ddim10': dict(files=list(files), draws=list(draws), sum_x=np.ones((2, 1, 2, 2)) * K,
                               sum_x2=np.ones(2) * K, err=np.ones((K, 2)))}
    m = db.merge_shards([shard([0, 1]), shard([2])])
    assert m['ddim10']['draws'] == [0, 1, 2] and np.allclose(m['ddim10']['sum_x'], 3) and m['ddim10']['err'].shape == (3, 2)
    with pytest.raises(ValueError, match='two shards'):
        db.merge_shards([shard([0, 1]), shard([1])])
    with pytest.raises(ValueError, match='different cubes'):
        db.merge_shards([shard([0]), shard([1], files=('a', 'c'))])


def test_shard_roundtrip(tmp_path):
    rows = {'t999': dict(files=['MatFlower60/Train/P022/a_1.mat'], draws=[0, 1], sum_x=np.zeros((1, 2, 2, 2)),
                         sum_x2=np.zeros(1), err=np.zeros((2, 1)))}
    p = str(tmp_path / 's.npz')
    db.save_shard(p, rows, dict(device='cpu'))
    back, meta = db.load_shard(p)
    assert back['t999']['files'] == rows['t999']['files'] and back['t999']['draws'] == [0, 1]
    assert np.allclose(back['t999']['err'], 0) and meta == dict(device='cpu')


def test_outputs_refused_inside_the_results_tree_outside_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'RESULTS_ROOT', str(tmp_path / 'root'))
    monkeypatch.setattr(db, 'DIAG_DIR', str(tmp_path / 'root' / 'diagnostics'))
    with pytest.raises(SystemExit, match='results tree'):
        db.check_out(str(tmp_path / 'root' / 'eval' / 'x.json'))
    db.check_out(str(tmp_path / 'root' / 'diagnostics' / 'x.json'))
    db.check_out(str(tmp_path / 'elsewhere.json'))


def test_consistency_reads_the_stored_seed_row_by_file():
    ev_b = {'rows': [dict(name='ddim1', files=['a', 'b'], seeds=[0, 1], per_item={'rmse_pct': [[1.0, 2.0], [3.0, 4.0]]})]}
    d = dict(files=['b', 'a'], draws=[1, 0, 5], err=np.array([[4.0, 3.0], [2.0, 1.5], [9.0, 9.0]]))
    assert db.consistency(d, ev_b, 't999') == {'1': 0.0, '0': 0.5}


def test_merge_end_to_end_on_synthetic_shards(tmp_path, monkeypatch):
    files = [f'MatFlower60/Train/P022/img{i}_1.mat' for i in range(4)]
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.5, 0.5, (4, 2, 2, 2))
    b = np.array([3.0, 3.2, 2.8, 3.1])

    class FakeDS:
        img_list = [os.path.join(str(tmp_path), f) for f in files]

        def __getitem__(self, i):
            return x[i].transpose(1, 2, 0), None, None

        def __len__(self):
            return 4
    monkeypatch.setattr(db, 'dataset_for', lambda root, prior, d, fl: FakeDS())

    def shard(off):
        xs = np.stack([x + off + 0.01 * rng.standard_normal(x.shape) for _ in range(2)])
        return dict(files=files, draws=[0, 1], sum_x=xs.sum(0), sum_x2=(xs ** 2).reshape(2, 4, -1).mean(2).sum(0),
                    err=50 * np.sqrt(((xs - x[None]) ** 2).reshape(2, 4, -1).mean(2)))
    rows = {'t999': shard(0.02), 'ddim10': shard(0.04)}
    db.save_shard(str(tmp_path / 's.npz'), rows, dict(device='cuda:0'))
    stored = [dict(name=n, files=files, seeds=[0, 1], per_item={'rmse_pct': rows[r]['err'].tolist()})
              for n, r in (('ddim1', 't999'), ('ddim10', 'ddim10'))]
    (tmp_path / 'b.json').write_text(json.dumps(dict(baseline_files=files, baseline_per_item={'rmse_pct': list(b)},
                                                     rows=stored)))
    db.main(['merge', '--inputs', str(tmp_path / 's.npz'), '--b_eval', str(tmp_path / 'b.json'), '--data_root',
             str(tmp_path), '--prior', 'unused', '--out', str(tmp_path / 'o.json')])
    out = json.load(open(tmp_path / 'o.json'))['rows']
    assert 'lambda_vs_one_step_centre' in out['ddim10'] and 'lambda_vs_one_step_centre' not in out['t999']
    assert out['t999']['consistency_vs_stored'] == {'0': 0.0, '1': 0.0}
    c1 = db.row_stats(rows['t999']['sum_x'], rows['t999']['sum_x2'], rows['t999']['err'], x)['c2']   # H1 centre energy
    s10 = db.row_stats(rows['ddim10']['sum_x'], rows['ddim10']['sum_x2'], rows['ddim10']['err'], x)
    r10 = out['ddim10']
    assert r10['pred_h1'] == pytest.approx(float(np.mean(np.sqrt(np.clip(c1 + (s10['o'] - c1) / 2, 0, None)))))
    assert r10['lambda_vs_one_step_centre']['value'] == pytest.approx(db.centre_lambda(b, s10['c2'], c1))
    assert r10['calibration']['ratio_of_means'] == pytest.approx(float(s10['spread'].mean() / c1.mean()))
    assert r10['calibration']['per_cube_slope_on_b2'] == pytest.approx(float(np.polyfit(b ** 2, s10['spread'], 1)[0]))


def test_merge_needs_the_t999_row(tmp_path):
    rows = {'ddim10': dict(files=['MatFlower60/Train/P022/a_1.mat'], draws=[0, 1], sum_x=np.zeros((1, 1, 2, 2)),
                           sum_x2=np.zeros(1), err=np.zeros((2, 1)))}
    db.save_shard(str(tmp_path / 's.npz'), rows, dict(device='cuda:0'))
    with pytest.raises(ValueError, match='t999'):
        db.main(['merge', '--inputs', str(tmp_path / 's.npz'), '--b_eval', str(tmp_path / 'missing.json'),
                 '--out', str(tmp_path / 'o.json')])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_diagnose_b.py -q`
Expected: collection error `ImportError: cannot import name 'diagnose_b' from 'scripts'`.

- [ ] **Step 3: Write the script**

`scripts/diagnose_b.py`:

```python
"""Step-0 diagnostics of the trained residual diffusion model B (Phase 1a-prime, Task 3).

Is B's sampling chain centred on r = 0 (H0: it samples an unconditional residual) or on its one-step estimate (H1: a
conditional posterior sampler)? Is its spread calibrated to its one-step error? Does averaging B's t = 999 one-step
output over K noise draws beat a single draw? (Phase 1a RESULTS note, Decision step 0.)

  sample  reconstructs every cube of the requested rows for each requested draw k with the harness's own seeding
          (torch.manual_seed(k * 1000 + batch index), batch 8, dataset order) -- so draws 0 and 1 reproduce B's
          stored evaluation when sampled on CUDA (the sampler draws its noise from the device's generator; a CPU
          shard draws independent noise) -- and saves per-cube running sums to an .npz shard.
  merge   adds shards with disjoint draws (the t999 row is required), reloads the cubes and B's evaluation, and
          writes the statistics JSON.

Rows: ddim10 and t999 (the ddim1 sampler: one call at t = 999) on the 200-cube subset; ddpm1000 on the 50-cube DDPM
subset, as the harness scores it.

  python scripts/diagnose_b.py sample --rows ddim10 t999 --draws 0 1 2 3 4 5 6 7 --device cuda:0 --out <diag>/b_fast.npz
  python scripts/diagnose_b.py sample --rows ddpm1000 --draws 0 1 --device cuda:0 --out <diag>/b_ddpm_a.npz
  python scripts/diagnose_b.py sample --rows ddpm1000 --draws 2 3 --device cuda:1 --out <diag>/b_ddpm_b.npz
  python scripts/diagnose_b.py merge --inputs <diag>/b_fast.npz <diag>/b_ddpm_a.npz <diag>/b_ddpm_b.npz \
      --out <diag>/b_diagnostics.json

Energies are in pct² of the [0, 1] display scale, E = 2500 * mean((x_hat - x)^2), so sqrt(E) is the harness's RMSE %.
Per cube, with K draws x_k, their mean x_bar and the truth x:
  o      = mean_k E(x_k)                                   single-draw error energy
  ebar   = E(x_bar)                                        error energy of the K-draw mean
  spread = 2500 * sum_k mean((x_k - x_bar)^2) / (K - 1)    unbiased spread energy (per-element variance across draws)
  c2     = ebar - spread / K                               error energy of the chain's centre (unbiased)
The one-step centre energy c1 is the t999 row's c2: the error energy of the noise-free centre of B's t = 999 output.
(A single ddim1 draw's RMSE^2 also holds that output's own noise energy, so it is a biased H1 endpoint.)
lambda = (mean b^2 - mean c2) / (mean b^2 - mean c1): 0 = centred on r = 0 (b: the baseline's RMSE), 1 = centred on
the one-step centre; SE by a jackknife over source images.
calibration: ratio_of_means = mean spread / mean c1 and self_ratio = mean spread / mean c2 are both 1 for a calibrated
posterior sampler; per_cube_slope_on_b2 (spread regressed on the noise-free b^2) is then close to the reference slope
of c1 on b^2, while a slope near 0 means a cube-independent spread whatever the ratios.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import DATA_ROOT, EVAL_DIR, PRIOR, RESULTS_ROOT, file_sha256, git_head  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from scripts.eval_warmstart import (_mean_se, build_model, check_meta, dataset_for, load_subset,  # noqa: E402
                                    make_eval_loader, metrics, reconstruct, sampler_grid)
from scripts.paired_stats import cluster_se, source_image  # noqa: E402

B_CKPT = os.path.join(RESULTS_ROOT, 'runs', 'residual_d4_b64', 'checkpoint_epoch_100.pth')
B_EVAL = os.path.join(EVAL_DIR, 'residual', 'd4', 'residual_d4_b64__checkpoint_epoch_100.json')
DIAG_DIR = os.path.join(RESULTS_ROOT, 'diagnostics')
ROWS = {'ddim10': ('ddim10', 'files'), 't999': ('ddim1', 'files'), 'ddpm1000': ('ddpm1000', 'ddpm1000_subset')}
SCALE = 2500.0      # E = 2500 * mean squared error on [-1, 1] = (RMSE % of the [0, 1] scale)^2


def under(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def check_out(path):
    if under(path, RESULTS_ROOT) and not under(path, DIAG_DIR):
        raise SystemExit(f'{path} is inside the results tree {RESULTS_ROOT}; diagnostics write there only under {DIAG_DIR}')


def row_cfg(row):
    return next(c for c in sampler_grid('full') if c['name'] == ROWS[row][0])


def accumulate(batches, recon, draws, n_cubes):
    """batches: the cubes in order as (x, y_d, x0_hat) tensors; recon(batch, seed) -> x_hat on [-1, 1]. For each draw k
    and batch index bi calls recon(batch, k * 1000 + bi) (the harness's seeding) and returns per-cube sums: sum_x
    [N, C, H, W] float64, sum_x2 [N] (sum over draws of the per-element mean of x_hat^2), err [len(draws), N] RMSE %."""
    sum_x, sum_x2, err = None, np.zeros(n_cubes), np.zeros((len(draws), n_cubes))
    for di, k in enumerate(draws):
        i0 = 0
        for bi, batch in enumerate(batches):
            x_hat = recon(batch, k * 1000 + bi)
            xh = x_hat.detach().double().cpu().numpy()
            n = xh.shape[0]
            if i0 + n > n_cubes:
                raise ValueError(f'batches hold more than the expected {n_cubes} cubes')
            if sum_x is None:
                sum_x = np.zeros((n_cubes,) + xh.shape[1:])
            sum_x[i0:i0 + n] += xh
            sum_x2[i0:i0 + n] += (xh ** 2).reshape(n, -1).mean(1)
            err[di, i0:i0 + n] = metrics(x_hat.detach().cpu().double(), batch[0].cpu().double())['rmse_pct'].numpy()
            i0 += n
        if i0 != n_cubes:
            raise ValueError(f'batches hold {i0} cubes, expected {n_cubes}')
    return sum_x, sum_x2, err


def row_stats(sum_x, sum_x2, err, x):
    """Per-cube energies from K draws (see the module docstring). x: [N, C, H, W] truth on [-1, 1]."""
    K = err.shape[0]
    if K < 2:
        raise ValueError('need at least 2 draws for a spread estimate')
    xbar = sum_x / K
    ebar = SCALE * ((xbar - x) ** 2).reshape(len(x), -1).mean(1)
    spread = SCALE * (sum_x2 - K * (xbar ** 2).reshape(len(x), -1).mean(1)) / (K - 1)
    return dict(K=int(K), o=(err ** 2).mean(0), ebar=ebar, spread=spread, c2=ebar - spread / K,
                kmean_rmse=np.sqrt(ebar), single_rmse=err.mean(0))


def centre_lambda(b, c2, c2_one_step):
    """(mean b^2 - mean c2) / (mean b^2 - mean c2_one_step), all in pct^2: 0 = the chain is centred on r = 0, 1 = on
    the one-step centre (c2_one_step: the t999 row's per-cube c2, the noise-free error energy of B's one-step output)."""
    b2 = float(np.mean(np.asarray(b) ** 2))
    return (b2 - float(np.mean(c2))) / (b2 - float(np.mean(c2_one_step)))


def jackknife_se(stat, files):
    """Leave-one-source-image-out jackknife SE of stat(mask) over the cubes `files`; None with fewer than 2 images."""
    groups = sorted({source_image(f) for f in files})
    if len(groups) < 2:
        return None
    gid = np.array([groups.index(source_image(f)) for f in files])
    vals = np.array([stat(gid != g) for g in range(len(groups))])
    G = len(groups)
    return float(np.sqrt((G - 1) / G * ((vals - vals.mean()) ** 2).sum()))


def merge_shards(shards):
    """Add shards row by row; their draws must be disjoint and their cube lists identical."""
    out = {}
    for sh in shards:
        for row, d in sh.items():
            if row not in out:
                out[row] = dict(files=list(d['files']), draws=list(d['draws']), sum_x=d['sum_x'].copy(),
                                sum_x2=d['sum_x2'].copy(), err=d['err'].copy())
                continue
            o = out[row]
            if list(d['files']) != o['files']:
                raise ValueError(f'{row}: shards score different cubes')
            both = sorted(set(d['draws']) & set(o['draws']))
            if both:
                raise ValueError(f'{row}: draws {both} appear in two shards')
            o['draws'] += list(d['draws'])
            o['sum_x'] += d['sum_x']
            o['sum_x2'] += d['sum_x2']
            o['err'] = np.concatenate([o['err'], d['err']])
    return out


def save_shard(path, rows, meta):
    arrays = {'meta': np.array(json.dumps(meta))}
    for row, d in rows.items():
        arrays[f'{row}.files'] = np.array(d['files'])
        arrays[f'{row}.draws'] = np.array(d['draws'])
        arrays[f'{row}.sum_x'] = d['sum_x']
        arrays[f'{row}.sum_x2'] = d['sum_x2']
        arrays[f'{row}.err'] = d['err']
    tmp = path + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_shard(path):
    z = np.load(path)
    rows = {}
    for key in z.files:
        if key != 'meta':
            row, field = key.split('.', 1)
            rows.setdefault(row, {})[field] = z[key]
    for d in rows.values():
        d['files'] = [str(f) for f in d['files']]
        d['draws'] = [int(k) for k in d['draws']]
    return rows, json.loads(str(z['meta']))


def _ms(v):
    m, se = _mean_se(np.asarray(v, dtype=float))
    return dict(mean=float(m), se=float(se))


def _pair(d, files):
    se_cluster, n_clusters = cluster_se(d, files)
    return dict(_ms(d), se_cluster=se_cluster, n_clusters=n_clusters)


STORED = {'ddim10': 'ddim10', 't999': 'ddim1', 'ddpm1000': 'ddpm1000'}


def consistency(d, ev_b, row):
    """Max |RMSE %| difference between this run's draws 0 / 1 and B's stored evaluation (same seeds, same batching)."""
    r = next(x for x in ev_b['rows'] if x['name'] == STORED[row])
    col = {f: i for i, f in enumerate(r['files'])}
    out = {}
    for di, k in enumerate(d['draws']):
        if k in r['seeds']:
            s = r['per_item']['rmse_pct'][r['seeds'].index(k)]
            out[str(k)] = float(max(abs(d['err'][di, i] - s[col[f]]) for i, f in enumerate(d['files'])))
    return out


def cmd_sample(args):
    check_out(args.out)
    device = torch.device(args.device)
    trainer = DiffusionTrainer(device=device, prediction_type='v', loss_type='l1', snr_gamma=None)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model, rows, t0 = None, {}, time.time()
    for row in args.rows:
        ds = dataset_for(args.data_root, args.prior, args.d, load_subset(args.data_root, ROWS[row][1]))
        if model is None:
            check_meta(ck, SimpleNamespace(ckpt=args.checkpoint, mode='residual', d=args.d, base_channels=args.base_channels,
                                           prediction_type='v'), ds.sigma_d, file_sha256(args.prior))
            model = build_model('residual', args.base_channels, device)
            model.load_state_dict(ck.get('model_state_dict', ck), strict=True)
        batches = list(make_eval_loader(ds, args.batch_size))
        cfg, sigma_d = row_cfg(row), ds.sigma_d
        sum_x, sum_x2, err = accumulate(
            batches, lambda batch, seed: reconstruct(model, trainer, 'residual', batch, cfg, sigma_d, seed),
            args.draws, len(ds))
        rows[row] = dict(files=[os.path.relpath(f, args.data_root) for f in ds.img_list], draws=list(args.draws),
                         sum_x=sum_x, sum_x2=sum_x2, err=err)
        print(f'{row}: {len(ds)} cubes x {len(args.draws)} draws, mean single-draw RMSE {err.mean():.4f} % '
              f'({time.time() - t0:.0f} s)', flush=True)
    meta = dict(checkpoint=os.path.abspath(args.checkpoint), checkpoint_sha256=file_sha256(args.checkpoint),
                prior_sha256=file_sha256(args.prior), device=str(device), batch_size=args.batch_size,
                git_head=git_head(), torch=torch.__version__, seconds=time.time() - t0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_shard(args.out, rows, meta)
    print('wrote', args.out)


def cmd_merge(args):
    check_out(args.out)
    loaded = [load_shard(p) for p in args.inputs]
    non_cuda = [m.get('device') for _, m in loaded if not str(m.get('device', '')).startswith('cuda')]
    if non_cuda:
        print(f'WARNING: shards sampled on {non_cuda}: the sampler noise comes from the device generator, so draws 0/1 '
              f'cannot reproduce the GPU-made evaluation; consistency_vs_stored is not a seeding check for them', flush=True)
    rows = merge_shards([sh for sh, _ in loaded])
    if 't999' not in rows:
        raise ValueError('merge needs the t999 row: its per-cube c2 is the one-step centre energy (the H1 endpoint)')
    ev_b = json.load(open(args.b_eval))
    base = dict(zip(ev_b['baseline_files'], ev_b['baseline_per_item']['rmse_pct']))
    d1 = next(r for r in ev_b['rows'] if r['name'] == 'ddim1')
    m1 = dict(zip(d1['files'], np.asarray(d1['per_item']['rmse_pct'], dtype=float).mean(0)))
    out, one_step_c2 = {}, None
    for row in sorted(rows, key=lambda r: r != 't999'):            # t999 first: its c2 is the one-step centre energy
        d = rows[row]
        files = d['files']
        ds = dataset_for(args.data_root, args.prior, args.d, [os.path.join(args.data_root, f) for f in files])
        if [os.path.relpath(f, args.data_root) for f in ds.img_list] != files:
            raise ValueError(f'{row}: the dataset order differs from the shard order')
        x = np.stack([ds[i][0] for i in range(len(ds))]).transpose(0, 3, 1, 2).astype(np.float64)
        s = row_stats(d['sum_x'], d['sum_x2'], d['err'], x)
        b = np.array([base[f] for f in files])
        if row == 't999':
            one_step_c2 = dict(zip(files, s['c2']))
        c1 = np.array([one_step_c2[f] for f in files])           # H1 centre energy per cube (pct^2)
        ms = np.array([m1[f] for f in files])                    # B's stored ddim1 RMSE, mean of its 2 seeds

        def lam(mask):
            return centre_lambda(b[mask], s['c2'][mask], c1[mask])

        def lam_single(mask):
            return centre_lambda(b[mask], s['c2'][mask], ms[mask] ** 2)

        def slope(mask):
            return float(np.polyfit(b[mask] ** 2, s['spread'][mask], 1)[0])
        every = np.ones(len(files), bool)
        rec = dict(K=s['K'], draws=sorted(d['draws']), n=len(files), baseline_rmse=float(b.mean()),
                   ddim1_rmse=float(ms.mean()), single_rmse=_ms(s['single_rmse']), kmean_rmse=_ms(s['kmean_rmse']),
                   kmean_minus_single=_pair(s['kmean_rmse'] - s['single_rmse'], files),
                   mean_o=float(s['o'].mean()), mean_ebar=float(s['ebar'].mean()),
                   mean_spread=float(s['spread'].mean()), mean_c2=float(s['c2'].mean()),
                   mean_c2_one_step=float(c1.mean()),
                   pred_h0=float(np.mean(np.sqrt(np.clip(b ** 2 + (s['o'] - b ** 2) / s['K'], 0, None)))),
                   pred_h1=float(np.mean(np.sqrt(np.clip(c1 + (s['o'] - c1) / s['K'], 0, None)))),
                   calibration=dict(ratio_of_means=float(s['spread'].mean() / c1.mean()),
                                    self_ratio=float(s['spread'].mean() / s['c2'].mean()),
                                    per_cube_slope_on_b2=slope(every), per_cube_slope_se=jackknife_se(slope, files),
                                    reference_slope_c2_one_step_on_b2=float(np.polyfit(b ** 2, c1, 1)[0]),
                                    per_cube_corr=float(np.corrcoef(b ** 2, s['spread'])[0, 1])),
                   consistency_vs_stored=consistency(d, ev_b, row))
        if row != 't999':                                        # for t999 itself lambda is 1 by construction
            rec['lambda_vs_one_step_centre'] = dict(value=lam(every), se_cluster=jackknife_se(lam, files))
        rec['lambda_vs_ddim1_single_draw'] = dict(
            value=lam_single(every), se_cluster=jackknife_se(lam_single, files),
            note="reference only: a single ddim1 draw's RMSE^2 includes the one-step output's own noise energy, so a "
                 "chain centred on the one-step centre reads above 1; for the t999 row this value measures that bias")
        rec['per_cube'] = dict(files=files, single_rmse=s['single_rmse'].tolist(), kmean_rmse=s['kmean_rmse'].tolist(),
                               spread=s['spread'].tolist(), c2=s['c2'].tolist())
        out[row] = rec
        cal = rec['calibration']
        lam_txt = (f"lambda {rec['lambda_vs_one_step_centre']['value']:.3f} "
                   if 'lambda_vs_one_step_centre' in rec else '')
        print(f"{row}: K={s['K']} single {rec['single_rmse']['mean']:.4f} K-mean {rec['kmean_rmse']['mean']:.4f} "
              f"(pred H0 {rec['pred_h0']:.4f}, H1 {rec['pred_h1']:.4f}) {lam_txt}calibration {cal['ratio_of_means']:.3f} "
              f"slope {cal['per_cube_slope_on_b2']:.3f} (reference {cal['reference_slope_c2_one_step_on_b2']:.3f}) "
              f"consistency {rec['consistency_vs_stored']}", flush=True)
    result = dict(provenance=dict(inputs=[os.path.abspath(p) for p in args.inputs], shards=[m for _, m in loaded],
                                  b_eval=os.path.abspath(args.b_eval), git_head=git_head()), rows=out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, args.out)
    print('wrote', args.out)


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--checkpoint', default=B_CKPT)
    s.add_argument('--rows', nargs='+', choices=list(ROWS), required=True)
    s.add_argument('--draws', type=int, nargs='+', required=True)
    s.add_argument('--device', default='cuda:0')
    s.add_argument('--batch_size', type=int, default=8, help='8 reproduces the stored evaluation (seed per batch index)')
    s.add_argument('--base_channels', type=int, default=64)
    s.add_argument('--d', type=int, default=4)
    s.add_argument('--data_root', default=DATA_ROOT)
    s.add_argument('--prior', default=PRIOR)
    s.add_argument('--out', required=True)
    m = sub.add_parser('merge')
    m.add_argument('--inputs', nargs='+', required=True)
    m.add_argument('--b_eval', default=B_EVAL)
    m.add_argument('--d', type=int, default=4)
    m.add_argument('--data_root', default=DATA_ROOT)
    m.add_argument('--prior', default=PRIOR)
    m.add_argument('--out', required=True)
    return ap


def main(argv=None):
    args = get_parser().parse_args(argv)
    (cmd_sample if args.cmd == 'sample' else cmd_merge)(args)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_diagnose_b.py -q`, then the full suite.
Expected: 11 passed; full suite 234 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnose_b.py tests/test_diagnose_b.py
git commit -F <msgfile> -- scripts/diagnose_b.py tests/test_diagnose_b.py
```
Message: `feat(warmstart): step-0 diagnostics of B — K draws with the harness seeding, chain centre (lambda vs the noise-free one-step centre) and per-cube spread calibration with jackknife SEs` + trailers.

---

### Task 4: Gate `scripts/gate_1a_prime.py`

**Files:**
- Create: `scripts/gate_1a_prime.py`
- Test: `tests/test_gate_1a_prime.py`

**Interfaces:**
- Consumes: from `scripts/gate_1a.py` — `YSWAP_THRESHOLD`, `_need`, `_paired`, `_prov`, `baseline_by_file`, `check_done`, `ckpt_epoch`, `paired_vs_baseline`, `per_cube`, `row_named`. From `data_loader/warmstart_paths.py` — `EVAL_DIR`, `GATE_DEFAULT_K`.
- Produces:
  - `ONE_STEP_T = 999`; `proposed_validity(ys, k)`, which refuses a record at any other single t and returns `valid`, `invalid_reason` ('a', 'b', 'ab' or None), `spec_ratio` and `spec_rule_passes`;
  - `paired_by_file(row_x, row_y)`;
  - `check_inputs(regs, ev_b, require_done)`;
  - `judge(ev, ev_b, k)`;
  - `vs_b_kmean(ev, b_diag, k)`, report only: R minus the mean of B's K t = 999 draws (`b_diagnostics.json` rows.t999: `K`, `per_cube.files`, `per_cube.kmean_rmse` from Task 3);
  - `gate(regs, ev_b, b_yswap=None, k=2.0, require_done=True, b_diag=None) -> dict(k, require_done, models[...], b{...}, pairs[...], best, best_verdict, go_1b)`. `best` is the lowest-RMSE `PASS` model (the lowest-RMSE model when none passes); a seed pair's `go_1b` needs both models `PASS`, `target_met` and `spec_rule_passes`;
  - `out_name(run_names, require_done)`;
  - `main(argv)`, which writes `gate_1a_prime__<run>+<run>[__nodone].json` next to the first `--reg`.

- [ ] **Step 1: Write the failing tests**

`tests/test_gate_1a_prime.py`:

```python
"""Phase 1a-prime gate (Task 4): proposed one-step validity rule, verdicts, refusals, loss and seed pairs, best, the
report-only comparison with B's K-draw mean, CLI."""
import json

import numpy as np
import pytest

from scripts import gate_1a_prime as gp

FILES = [f'MatFlower60/Train/P0{22 + i % 3}/img{i // 2}_{i % 2}.mat' for i in range(12)]
BASE = np.linspace(2.0, 3.0, 12)
SPREAD = np.array([-0.05, 0.03, -0.02, 0.04, -0.01, 0.02, -0.03, 0.01, 0.0, 0.05, -0.04, 0.0])


def row(name, vals):
    vals = np.asarray(vals, float)
    d = vals - BASE
    return dict(name=name, nfe=1, files=list(FILES), rmse_pct=float(vals.mean()),
                rmse_pct_se=float(vals.std(ddof=1) / np.sqrt(12)), per_item={'rmse_pct': [list(vals), list(vals)]},
                paired_rmse_diff=float(d.mean()), paired_rmse_se=float(d.std(ddof=1) / np.sqrt(12)))


def yswap(true, swap):
    return {'t999': dict(true=float(np.mean(true)), swap_full=float(np.mean(swap))), 'passes': False,
            'per_cube': {'999': dict(true=list(true), swap_full=list(swap), swap_sensor_only=list(true), zero=list(swap))},
            'baseline_per_cube': list(BASE[:8]), 'files': FILES[:8]}


GOOD_YS = yswap(BASE[:8] - 0.15 + SPREAD[:8], BASE[:8] + 0.05 + 0.5 * SPREAD[:8])    # uses its condition, gain vanishes;
#                                                                                     spec 0.9 ratio fails (0.915)
TARGET_YS = yswap(BASE[:8] - 0.45 + SPREAD[:8], BASE[:8] + 0.05 + 0.5 * SPREAD[:8])  # the same at a target-size gain;
#                                                                                     spec 0.9 ratio passes (0.789)
BLIND_YS = yswap(BASE[:8] - 0.3 + SPREAD[:8], BASE[:8] - 0.3 + SPREAD[:8])           # swap changes nothing
STILL_YS = yswap(BASE[:8] - 0.3 + SPREAD[:8], BASE[:8] - 0.2 + SPREAD[:8])           # still beats the baseline swapped


def reg_eval(vals, ys, loss='l2', seed=42, ckpt='/r/regress_l2_d4_b64/checkpoint_epoch_100.pth', run='regress_l2_d4_b64',
             **prov):
    p = dict(mode='regress', d=4, base_channels=64, prediction_type='direct', prior_sha256='p' * 64, meta_checked=True,
             ckpt=ckpt, loss_type=loss, train_seed=seed)
    p.update(prov)
    return dict(provenance=p, grid='direct', run_name=run, rows=[row('direct', vals)], yswap=ys,
                baseline=dict(rmse_pct=float(BASE.mean())), baseline_files=list(FILES),
                baseline_per_item={'rmse_pct': list(BASE)})


def b_eval(vals, **prov):
    p = dict(mode='residual', d=4, base_channels=64, prediction_type='v', prior_sha256='p' * 64, meta_checked=True,
             ckpt='/r/residual_d4_b64/checkpoint_epoch_100.pth')
    p.update(prov)
    return dict(provenance=p, grid='full', run_name='residual_d4_b64', rows=[row('ddim1', vals)],
                baseline=dict(rmse_pct=float(BASE.mean())), baseline_files=list(FILES),
                baseline_per_item={'rmse_pct': list(BASE)})


B = b_eval(BASE - 0.1 + SPREAD)                                                       # B: 0.1 better than the baseline


def test_proposed_validity_three_cases():
    good = gp.proposed_validity(GOOD_YS)
    assert good['valid'] and good['t'] == 999 and good['n'] == 8 and not good['spec_rule_passes']
    assert good['invalid_reason'] is None and gp.proposed_validity(TARGET_YS)['spec_rule_passes']
    blind = gp.proposed_validity(BLIND_YS)
    assert not blind['valid'] and not blind['swap_full_minus_true']['pass'] and blind['invalid_reason'] == 'ab'
    still = gp.proposed_validity(STILL_YS)
    assert not still['valid'] and still['swap_full_minus_true']['pass'] and not still['swap_full_minus_baseline']['pass']
    assert still['invalid_reason'] == 'b'
    two_t = dict(GOOD_YS, per_cube={'100': GOOD_YS['per_cube']['999'], '300': GOOD_YS['per_cube']['999']})
    with pytest.raises(ValueError, match='single t'):
        gp.proposed_validity(two_t)


def test_proposed_validity_refuses_a_single_t_other_than_999():
    with pytest.raises(ValueError, match='t = 999'):
        gp.proposed_validity(dict(GOOD_YS, per_cube={'100': GOOD_YS['per_cube']['999']}))


def test_proposed_validity_part_b_tolerates_a_swap_within_k_se_below_the_baseline():
    near = yswap(BASE[:8] - 0.15 + SPREAD[:8], BASE[:8] - 0.005 + 0.5 * SPREAD[:8])
    v = gp.proposed_validity(near)
    b = v['swap_full_minus_baseline']
    assert b['diff'] < 0 <= b['diff'] + 2 * b['se'] and b['pass'] and v['valid']


def test_verdicts():
    m = gp.gate([reg_eval(BASE - 0.3 + SPREAD, GOOD_YS)], B, require_done=False)['models'][0]
    assert m['verdict'] == 'PASS' and m['invalid_reason'] is None
    m = gp.gate([reg_eval(BASE - 0.05 + 0.2 * SPREAD, GOOD_YS)], B, require_done=False)['models'][0]
    assert m['verdict'] == 'PASS_NOT_OVER_B' and m['vs_baseline']['pass'] and not m['vs_b_ddim1']['pass']
    assert gp.gate([reg_eval(BASE + 0.1 + SPREAD, GOOD_YS)], B, require_done=False)['models'][0]['verdict'] == 'FAIL'
    m = gp.gate([reg_eval(BASE - 0.3 + SPREAD, BLIND_YS)], B, require_done=False)['models'][0]
    assert m['verdict'] == 'INVALID' and m['verdict_if_valid'] == 'PASS' and m['invalid_reason'] == 'ab'
    m = gp.gate([reg_eval(BASE - 0.3 + SPREAD, STILL_YS)], B, require_done=False)['models'][0]
    assert m['verdict'] == 'INVALID' and m['invalid_reason'] == 'b'
    m = gp.gate([reg_eval(BASE - 0.4 + SPREAD, GOOD_YS)], B, require_done=False)['models'][0]
    assert m['target_met'] and m['target_rmse_pct'] == pytest.approx(0.85 * BASE.mean())


def test_pass_rule_is_diff_plus_k_iid_se():
    files = [f'MatFlower60/Train/P022/img{i // 2}_{i % 2}.mat' for i in range(12)]       # 6 images x 2 patches
    noise = np.repeat([0.03, -0.03, 0.02, -0.02, 0.01, -0.01], 2)                       # patches of an image move together

    def relabel(ev):
        ev = json.loads(json.dumps(ev))
        ev['baseline_files'] = list(files)
        for r in ev['rows']:
            r['files'] = list(files)
        return ev
    b = relabel(B)
    verdicts = {}
    for delta in (-0.016, -0.005):            # iid: -0.016 + 2 * 0.0065 < 0 (pass); the clustered SE 0.0097 would fail it
        m = gp.gate([relabel(reg_eval(BASE + delta + noise, GOOD_YS))], b, require_done=False)['models'][0]
        assert m['vs_baseline']['se_cluster'] > 1.3 * m['vs_baseline']['se']
        verdicts[delta] = m['vs_baseline']['pass']
    assert verdicts == {-0.016: True, -0.005: False}            # -0.005 < 0 but within 2 SE: a bare sign rule passes it


def test_b_valid_but_not_better_than_the_baseline_is_fail():
    g = gp.gate([reg_eval(BASE - 0.3 + SPREAD, GOOD_YS)], b_eval(BASE + 0.05 + SPREAD), {'yswap': GOOD_YS},
                require_done=False)
    assert g['b']['validity']['valid'] and g['b']['verdict_under_proposed_rule'] == 'FAIL'


def test_refusals():
    def good(**p):
        return reg_eval(BASE - 0.3 + SPREAD, GOOD_YS, **p)
    for bad in (dict(mode='residual'), dict(prediction_type='v'), dict(meta_checked=False), dict(prior_sha256='q' * 64),
                dict(d=2)):
        with pytest.raises(ValueError):
            gp.gate([good(**bad)], B, require_done=False)
    wrong_grid = good()
    wrong_grid['grid'] = 'full'
    with pytest.raises(ValueError, match='direct'):
        gp.gate([wrong_grid], B, require_done=False)
    shifted = good()
    shifted['baseline_per_item'] = {'rmse_pct': list(BASE + 0.01)}
    with pytest.raises(ValueError, match='baselines differ'):
        gp.gate([shifted], B, require_done=False)
    with pytest.raises(ValueError, match='different epochs'):
        gp.gate([good(), good(ckpt='/r/x/checkpoint_epoch_90.pth', loss_type='l1')], B, require_done=False)
    with pytest.raises(ValueError, match='full-grid residual'):
        gp.gate([good()], dict(B, grid='quick'), require_done=False)


def test_done_is_required(tmp_path):
    run, brun = tmp_path / 'regress_l2_d4_b64', tmp_path / 'residual_d4_b64'
    for d in (run, brun):
        d.mkdir()
        (d / 'DONE').write_text('100\n')
    b = b_eval(BASE - 0.1 + SPREAD, ckpt=str(brun / 'checkpoint_epoch_100.pth'))
    ok = reg_eval(BASE - 0.3 + SPREAD, GOOD_YS, ckpt=str(run / 'checkpoint_epoch_100.pth'))
    assert gp.gate([ok], b)['models'][0]['verdict'] == 'PASS'
    (run / 'DONE').write_text('90\n')
    with pytest.raises(ValueError, match='not the final checkpoint'):
        gp.gate([ok], b)


def test_loss_pair_seed_pair_and_best():
    l1 = reg_eval(BASE - 0.3 + SPREAD, GOOD_YS, loss='l1', run='regress_l1_d4_b64')
    l2 = reg_eval(BASE - 0.4 + SPREAD, TARGET_YS, loss='l2', run='regress_l2_d4_b64')
    g = gp.gate([l1, l2], B, require_done=False)
    assert g['pairs'][0]['kind'] == 'loss_pair' and g['pairs'][0]['a_minus_b']['diff'] == pytest.approx(0.1)
    assert g['best'] == 'regress_l2_d4_b64' and g['best_verdict'] == 'PASS' and g['go_1b'] is False
    s43 = reg_eval(BASE - 0.45 + SPREAD, TARGET_YS, loss='l2', seed=43, run='regress_l2_s43_d4_b64')
    g = gp.gate([l2, s43], B, require_done=False)
    assert g['pairs'][0]['kind'] == 'seed_pair' and g['pairs'][0]['go_1b'] is True and g['go_1b'] is True
    low_ratio = [reg_eval(BASE - 0.4 + SPREAD, GOOD_YS, loss='l2', run='regress_l2_d4_b64'),
                 reg_eval(BASE - 0.45 + SPREAD, GOOD_YS, loss='l2', seed=43, run='regress_l2_s43_d4_b64')]
    g = gp.gate(low_ratio, B, require_done=False)       # PASS and target met, but the spec 0.9 ratio fails at t = 999
    assert [m['verdict'] for m in g['models']] == ['PASS', 'PASS'] and all(m['target_met'] for m in g['models'])
    assert g['go_1b'] is False
    blind_l2 = reg_eval(BASE - 0.4 + SPREAD, BLIND_YS, loss='l2', run='regress_l2_d4_b64')
    g = gp.gate([l1, blind_l2], B, require_done=False)  # the lower-RMSE L2 is INVALID: best is the passing L1
    assert g['best'] == 'regress_l1_d4_b64' and g['best_verdict'] == 'PASS'


def test_seed_pair_needs_both_targets_met():
    l2 = reg_eval(BASE - 0.4 + SPREAD, TARGET_YS, loss='l2', run='regress_l2_d4_b64')
    s43 = reg_eval(BASE - 0.3 + SPREAD, TARGET_YS, loss='l2', seed=43, run='regress_l2_s43_d4_b64')   # 2.2 > 2.125
    g = gp.gate([l2, s43], B, require_done=False)
    assert [m['verdict'] for m in g['models']] == ['PASS', 'PASS'] and not g['models'][1]['target_met']
    assert g['pairs'][0]['kind'] == 'seed_pair' and g['pairs'][0]['go_1b'] is False and g['go_1b'] is False


def test_vs_b_t999_kmean_is_reported_but_not_binding():
    diag = {'rows': {'t999': {'K': 8, 'per_cube': {'files': list(FILES), 'kmean_rmse': list(BASE - 0.35 + SPREAD)}}}}
    reg = reg_eval(BASE - 0.3 + SPREAD, GOOD_YS)
    m = gp.gate([reg], B, require_done=False, b_diag=diag)['models'][0]
    assert m['verdict'] == 'PASS' and m['vs_b_t999_kmean']['nfe'] == 8          # B's 8-draw mean is 0.05 better
    assert m['vs_b_t999_kmean']['diff'] == pytest.approx(0.05) and not m['vs_b_t999_kmean']['pass']
    assert gp.gate([reg], B, require_done=False)['models'][0]['vs_b_t999_kmean'] is None
    diag['rows']['t999']['per_cube']['kmean_rmse'] = list(BASE - 0.3 + 2 * SPREAD + 0.002)    # R 0.002 better, within 2 SE
    vk = gp.gate([reg], B, require_done=False, b_diag=diag)['models'][0]['vs_b_t999_kmean']
    assert vk['diff'] == pytest.approx(-0.002) and vk['diff'] + 2 * vk['se'] > 0 and not vk['pass']
    diag['rows']['t999']['per_cube']['files'] = FILES[:11] + ['MatFlower60/Train/P022/other_1.mat']
    with pytest.raises(ValueError, match='different cubes'):
        gp.gate([reg], B, require_done=False, b_diag=diag)


def test_main_writes_the_named_record(tmp_path):
    pr, pb, pys, pd = tmp_path / 'reg.json', tmp_path / 'b.json', tmp_path / 'bys.json', tmp_path / 'diag.json'
    pr.write_text(json.dumps(reg_eval(BASE - 0.3 + SPREAD, GOOD_YS)))
    pb.write_text(json.dumps(B))
    pys.write_text(json.dumps({'yswap': GOOD_YS}))
    pd.write_text(json.dumps({'rows': {'t999': {'K': 8, 'per_cube': {'files': list(FILES),
                                                                     'kmean_rmse': list(BASE - 0.2 + SPREAD)}}}}))
    g = gp.main(['--reg', str(pr), '--b', str(pb), '--no_require_done'])
    out = tmp_path / 'gate_1a_prime__regress_l2_d4_b64__nodone.json'
    assert out.exists() and json.loads(out.read_text())['models'][0]['verdict'] == 'PASS'
    assert g['inputs']['b'] == str(pb) and g['b']['verdict_under_proposed_rule'] is None
    g = gp.main(['--reg', str(pr), '--b', str(pb), '--b_yswap', str(pys), '--b_diag', str(pd), '--no_require_done',
                 '--out', str(tmp_path / 'x.json')])
    assert g['b']['verdict_under_proposed_rule'] == 'PASS' and (tmp_path / 'x.json').exists()
    assert g['inputs']['b_diag'] == str(pd) and g['models'][0]['vs_b_t999_kmean']['pass']
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_gate_1a_prime.py -q`
Expected: collection error `ImportError: cannot import name 'gate_1a_prime' from 'scripts'`.

- [ ] **Step 3: Write the script**

`scripts/gate_1a_prime.py`:

```python
"""Phase 1a-prime gate: the one-step residual regressor(s) against the baseline and B's one-step read-out (ddim1) on
the same 200 cubes, under the PROPOSED one-step y-swap validity rule (plan deviation 1: binding for the verdicts in
this phase; the spec §7 0.9 ratio is reported, and go_1b requires it too). Torch-free.

  python scripts/gate_1a_prime.py \
      --reg <eval>/regress/d4/regress_l1_d4_b64__checkpoint_epoch_100__direct.json \
            <eval>/regress/d4/regress_l2_d4_b64__checkpoint_epoch_100__direct.json \
      --b_yswap <results>/diagnostics/yswap_residual_d4_b64_t999_n32.json \
      --b_diag <results>/diagnostics/b_diagnostics.json

Per regressor R: vs_baseline (R - baseline per cube) and vs_b_ddim1 (R - B's seed-mean ddim1 per cube) pass when
diff + k * SE < 0 (iid SE; the CR1 SE over source images is reported). Validity, on R's own y-swap at its single call
t = 999: (a) swap_full - true > k * SE and (b) swap_full - baseline + k * SE >= 0 (with a swapped condition the gain
over the baseline vanishes); invalid_reason names the failed part(s), 'a', 'b' or 'ab'. Verdict: INVALID unless valid;
else PASS (beats the baseline and B), PASS_NOT_OVER_B (beats the baseline only) or FAIL. target_met: R's RMSE <= 0.85 x
the baseline (the spec §8 1b bar). Report only (--b_diag): vs_b_t999_kmean, R minus the mean of B's K one-step draws
at t = 999 (K NFE). B is re-validated under the same rule from its t = 999 y-swap (--b_yswap). Two regressors with
different losses and the same training seed form a loss_pair; two with the same loss and different seeds form a
seed_pair, whose go_1b is true only when both are PASS (hence valid), target_met and pass the spec 0.9 ratio at
t = 999. best is the lowest-RMSE PASS regressor (the lowest-RMSE one when none passes); best_verdict is its verdict.
"""
import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import EVAL_DIR, GATE_DEFAULT_K as DEFAULT_K  # noqa: E402
from scripts.gate_1a import (YSWAP_THRESHOLD, _need, _paired, _prov, baseline_by_file, check_done, ckpt_epoch,  # noqa: E402
                             paired_vs_baseline, per_cube, row_named)

B_EVAL = os.path.join(EVAL_DIR, 'residual', 'd4', 'residual_d4_b64__checkpoint_epoch_100.json')
TARGET_FRACTION = 0.85          # spec §8: the 1b bar is 15 % below the baseline
ONE_STEP_T = 999                # REGRESS_T and B's ddim1 call (repeated here: the gate stays torch-free)


def proposed_validity(ys, k=DEFAULT_K):
    """The proposed rule on a y-swap record with per-cube values at the single one-step t (see the module docstring)."""
    per = _need(ys, 'per_cube', 'y-swap record')
    if len(per) != 1:
        raise ValueError(f'the proposed rule needs a y-swap record at a single t, got t = {sorted(per)}')
    (t, cubes), = per.items()
    if int(t) != ONE_STEP_T:
        raise ValueError(f'the proposed rule is defined at the one-step call t = {ONE_STEP_T}, got t = {t}')
    files = _need(ys, 'files', 'y-swap record')
    if files is None:
        raise ValueError('the y-swap record has no files (needed for the clustered SE)')
    true, swap = np.asarray(cubes['true'], float), np.asarray(cubes['swap_full'], float)
    base = np.asarray(_need(ys, 'baseline_per_cube', 'y-swap record'), float)
    a, b, g = _paired(swap - true, files), _paired(swap - base, files), _paired(true - base, files)
    a['pass'] = bool(a['diff'] - k * a['se'] > 0)
    b['pass'] = bool(b['diff'] + k * b['se'] >= 0)
    ratio = float(true.mean() / swap.mean())
    reason = ''.join(part for part, ok in (('a', a['pass']), ('b', b['pass'])) if not ok) or None
    return dict(t=int(t), n=len(files), swap_full_minus_true=a, swap_full_minus_baseline=b, true_minus_baseline=g,
                valid=reason is None, invalid_reason=reason, spec_ratio=ratio, spec_threshold=YSWAP_THRESHOLD,
                spec_rule_passes=bool(ratio <= YSWAP_THRESHOLD))


def paired_by_file(row_x, row_y):
    """row_x - row_y per cube (seed means), paired by file name; both rows must score the same cubes."""
    y = dict(zip(row_y['files'], per_cube(row_y)))
    if set(row_x['files']) != set(y):
        raise ValueError('rows score different cubes')
    return _paired(np.array([v - y[f] for f, v in zip(row_x['files'], per_cube(row_x))]), row_x['files'])


def check_inputs(regs, ev_b, require_done):
    for ev in regs:
        if _prov(ev, 'mode') != 'regress' or ev.get('grid') != 'direct' or _prov(ev, 'prediction_type') != 'direct':
            raise ValueError(f"--reg must be a direct-grid regress evaluation, got mode={_prov(ev, 'mode')!r} "
                             f"grid={ev.get('grid')!r} prediction_type={_prov(ev, 'prediction_type')!r}")
    if _prov(ev_b, 'mode') != 'residual' or ev_b.get('grid') != 'full':
        raise ValueError(f"--b must be B's full-grid residual evaluation, got mode={_prov(ev_b, 'mode')!r} "
                         f"grid={ev_b.get('grid')!r}")
    evs = list(regs) + [ev_b]
    for ev in evs:
        if not _prov(ev, 'meta_checked'):
            raise ValueError(f"{_prov(ev, 'ckpt')}: checkpoint meta was not checked")
    for key in ('d', 'base_channels', 'prior_sha256'):
        vals = {str(_prov(ev, key)) for ev in evs}
        if len(vals) != 1:
            raise ValueError(f'evaluations differ in provenance.{key}: {sorted(vals)}')
    epochs = {_prov(ev, 'ckpt'): ckpt_epoch(_prov(ev, 'ckpt')) for ev in regs}
    if len(set(epochs.values())) != 1:
        raise ValueError(f'the regressors stopped at different epochs: {epochs}')
    if require_done:
        for ev in evs:
            check_done(ev)
    base_b = baseline_by_file(ev_b)
    for ev in regs:
        base = baseline_by_file(ev)
        if base is None or set(base) != set(base_b) or max(abs(base[f] - base_b[f]) for f in base) > 1e-6:
            raise ValueError(f"{_prov(ev, 'ckpt')}: per-cube baselines differ from B's (different cubes or prior)")


def judge(ev, ev_b, k):
    rr, rb = row_named(ev, 'direct'), row_named(ev_b, 'ddim1')
    if rr is None or rb is None:
        raise ValueError("needs the regressor's 'direct' row and B's 'ddim1' row")
    vs_base, vs_b = paired_vs_baseline(rr, baseline_by_file(ev)), paired_by_file(rr, rb)
    for r in (vs_base, vs_b):
        r['pass'] = bool(r['diff'] + k * r['se'] < 0)
    validity = proposed_validity(_need(ev, 'yswap'), k)
    verdict_if_valid = 'PASS' if vs_base['pass'] and vs_b['pass'] else ('PASS_NOT_OVER_B' if vs_base['pass'] else 'FAIL')
    baseline = float(_need(ev, 'baseline')['rmse_pct'])
    prov = ev['provenance']
    return dict(run_name=ev.get('run_name'), ckpt=prov['ckpt'], epoch=ckpt_epoch(prov['ckpt']),
                loss_type=prov.get('loss_type'), train_seed=prov.get('train_seed'), rmse_pct=rr['rmse_pct'],
                rmse_pct_se=rr['rmse_pct_se'], baseline_rmse_pct=baseline, target_rmse_pct=TARGET_FRACTION * baseline,
                target_met=bool(rr['rmse_pct'] <= TARGET_FRACTION * baseline), vs_baseline=vs_base, vs_b_ddim1=vs_b,
                validity=validity, verdict_if_valid=verdict_if_valid, invalid_reason=validity['invalid_reason'],
                verdict=verdict_if_valid if validity['valid'] else 'INVALID')


def vs_b_kmean(ev, b_diag, k=DEFAULT_K):
    """Report only: R's direct row minus the mean of B's K one-step draws at t = 999 (K NFE), per cube, from
    b_diagnostics.json (scripts/diagnose_b.py merge)."""
    t9 = _need(_need(b_diag, 'rows', 'b_diagnostics'), 't999', 'b_diagnostics rows')
    km = dict(zip(t9['per_cube']['files'], t9['per_cube']['kmean_rmse']))
    rr = row_named(ev, 'direct')
    if set(rr['files']) != set(km):
        raise ValueError("b_diagnostics' t999 row scores different cubes from the regressor evaluation")
    r = _paired(np.array([v - km[f] for f, v in zip(rr['files'], per_cube(rr))]), rr['files'])
    r['pass'] = bool(r['diff'] + k * r['se'] < 0)
    r['nfe'] = int(t9['K'])
    return r


def gate(regs, ev_b, b_yswap=None, k=DEFAULT_K, require_done=True, b_diag=None):
    check_inputs(regs, ev_b, require_done)
    models = [judge(ev, ev_b, k) for ev in regs]
    for m, ev in zip(models, regs):
        m['vs_b_t999_kmean'] = None if b_diag is None else vs_b_kmean(ev, b_diag, k)
    rb = row_named(ev_b, 'ddim1')
    b = dict(rmse_pct=rb['rmse_pct'], vs_baseline=paired_vs_baseline(rb, baseline_by_file(ev_b)),
             validity=None if b_yswap is None else proposed_validity(_need(b_yswap, 'yswap'), k))
    b['vs_baseline']['pass'] = bool(b['vs_baseline']['diff'] + k * b['vs_baseline']['se'] < 0)
    b['verdict_under_proposed_rule'] = None if b['validity'] is None else (
        ('PASS' if b['vs_baseline']['pass'] else 'FAIL') if b['validity']['valid'] else 'INVALID')
    pairs = []
    for (i, mi), (j, mj) in combinations(enumerate(models), 2):
        if mi['loss_type'] != mj['loss_type'] and mi['train_seed'] == mj['train_seed']:
            kind = 'loss_pair'
        elif mi['loss_type'] == mj['loss_type'] and mi['train_seed'] != mj['train_seed']:
            kind = 'seed_pair'
        else:
            kind = 'other'
        rec = dict(kind=kind, a=mi['run_name'], b=mj['run_name'],
                   a_minus_b=paired_by_file(row_named(regs[i], 'direct'), row_named(regs[j], 'direct')))
        if kind == 'seed_pair':
            rec['go_1b'] = all(m['verdict'] == 'PASS' and m['target_met'] and m['validity']['spec_rule_passes']
                               for m in (mi, mj))
        pairs.append(rec)
    passing = [m for m in models if m['verdict'] == 'PASS']
    best = min(passing or models, key=lambda m: m['rmse_pct'])
    return dict(k=k, require_done=require_done, models=models, b=b, pairs=pairs, best=best['run_name'],
                best_verdict=best['verdict'], go_1b=any(p.get('go_1b', False) for p in pairs))


def out_name(run_names, require_done):
    return 'gate_1a_prime__' + '+'.join(run_names) + ('' if require_done else '__nodone') + '.json'


def print_report(g):
    for m in g['models']:
        v, vb, vv = m['vs_baseline'], m['vs_b_ddim1'], m['validity']
        print(f"{m['run_name']} ({m['loss_type']}, seed {m['train_seed']}, epoch {m['epoch']}): {m['verdict']}  "
              f"RMSE {m['rmse_pct']:.4f} ± {m['rmse_pct_se']:.4f} %  target {m['target_rmse_pct']:.4f} "
              f"{'met' if m['target_met'] else 'not met'}")
        print(f"   - baseline {v['diff']:+.4f} ± {v['se']:.4f} (clustered {v['se_cluster']:.4f})  "
              f"- B ddim1 {vb['diff']:+.4f} ± {vb['se']:.4f} (clustered {vb['se_cluster']:.4f})")
        if m['vs_b_t999_kmean'] is not None:
            vk = m['vs_b_t999_kmean']
            print(f"   - B t999 mean of {vk['nfe']} draws (report only) {vk['diff']:+.4f} ± {vk['se']:.4f} "
                  f"(clustered {vk['se_cluster']:.4f})")
        print(f"   validity at t = {vv['t']} on {vv['n']} cubes: swap_full - true {vv['swap_full_minus_true']['diff']:+.4f} ± "
              f"{vv['swap_full_minus_true']['se']:.4f}, swap_full - baseline {vv['swap_full_minus_baseline']['diff']:+.4f} ± "
              f"{vv['swap_full_minus_baseline']['se']:.4f} -> "
              f"{'valid' if vv['valid'] else 'INVALID (' + vv['invalid_reason'] + ')'} "
              f"(spec 0.9 ratio {vv['spec_ratio']:.3f})")
    b = g['b']
    print(f"B ddim1 {b['rmse_pct']:.4f} %  - baseline {b['vs_baseline']['diff']:+.4f} ± {b['vs_baseline']['se']:.4f}  "
          f"under the proposed rule: {b['verdict_under_proposed_rule']}")
    for p in g['pairs']:
        print(f"{p['kind']}: {p['a']} - {p['b']} {p['a_minus_b']['diff']:+.4f} ± {p['a_minus_b']['se']:.4f}"
              + (f"  go_1b {p['go_1b']}" if 'go_1b' in p else ''))
    print(f"best {g['best']} ({g['best_verdict']}); go_1b {g['go_1b']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--reg', nargs='+', required=True)
    ap.add_argument('--b', default=B_EVAL)
    ap.add_argument('--b_yswap', default=None, help="B's t = 999 y-swap (scripts/yswap_one_step.py --t 999 --n 32)")
    ap.add_argument('--b_diag', default=None, help="B's step-0 diagnostics (scripts/diagnose_b.py merge); adds the "
                                                   "report-only comparison with the mean of B's K one-step draws")
    ap.add_argument('--k', type=float, default=DEFAULT_K)
    ap.add_argument('--no_require_done', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    regs = [json.load(open(p)) for p in args.reg]
    ev_b = json.load(open(args.b))
    b_yswap = json.load(open(args.b_yswap)) if args.b_yswap else None
    b_diag = json.load(open(args.b_diag)) if args.b_diag else None
    g = gate(regs, ev_b, b_yswap, k=args.k, require_done=not args.no_require_done, b_diag=b_diag)
    g['inputs'] = dict(reg=[os.path.abspath(p) for p in args.reg], b=os.path.abspath(args.b),
                       b_yswap=os.path.abspath(args.b_yswap) if args.b_yswap else None,
                       b_diag=os.path.abspath(args.b_diag) if args.b_diag else None)
    print_report(g)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.reg[0])),
                                   out_name([m['run_name'] for m in g['models']], g['require_done']))
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(g, f, indent=1)
    os.replace(tmp, out)
    print('wrote', out)
    return g


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_gate_1a_prime.py -q`, then the full suite.
Expected: 12 passed; full suite 246 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/gate_1a_prime.py tests/test_gate_1a_prime.py
git commit -F <msgfile> -- scripts/gate_1a_prime.py tests/test_gate_1a_prime.py
```
Message: `feat(warmstart): gate 1a-prime — regressor vs baseline and B's ddim1 on the same cubes, proposed one-step y-swap validity at t = 999 with invalid_reason, loss and seed pairs (go_1b also needs the spec ratio), best = lowest-RMSE PASS, report-only comparison with B's K-draw mean` + trailers.

---

### Task 5: `launch_run.sh` RUN_NAME, regress smoke test, regress timing (GPU)

**Files:**
- Modify: `scripts/pilot/launch_run.sh` (the `RUN=` line and the usage comment)
- Create: `scripts/pilot/smoke_regress.sh`
- Test: `tests/test_launch_run_name.py` (new)
- Append (git-ignored): `RESULTS_ROOT/timing_phase1.json`

**Interfaces:**
- Consumes: Tasks 1–2; the existing `launch_run.sh` behaviour (busy check, lock, DONE refusal, launches counter, BLAS exports, `--resume auto`, extra args appended last).
- Produces: `RUN_NAME=<name> bash scripts/pilot/launch_run.sh regress <gpu> --prediction_type direct --loss_type l1|l2 [--seed N]` → a run directory `RESULTS_ROOT/runs/<name>`; a verified GPU path for training, resume, direct-grid evaluation and meta refusal; a timing record `regress_d4_b64_bs8x1_w8`.

- [ ] **Step 1: Write the failing test**

`tests/test_launch_run_name.py`:

```python
"""launch_run.sh honours RUN_NAME (Phase 1a-prime Task 5): the run directory is RUN_NAME and the extra arguments reach
the trainer last."""
import os
import subprocess
import time

from data_loader.warmstart_paths import REPO

LAUNCH = os.path.join(REPO, 'scripts', 'pilot', 'launch_run.sh')


def test_run_name_selects_the_run_directory_and_extra_args_pass_through(tmp_path):
    stub = tmp_path / 'stub'
    stub.mkdir()
    (stub / 'nvidia-smi').write_text('#!/bin/bash\ncase "$*" in *memory.free*) echo 20000 ;; esac\n')
    (stub / 'python').write_text('#!/bin/bash\necho "$*" >> "$STUB_ARGS"\n')
    for f in ('nvidia-smi', 'python'):
        os.chmod(stub / f, 0o755)
    args_file = tmp_path / 'stub_args'
    env = dict(os.environ, PATH=f'{stub}:{os.environ["PATH"]}', RESIDUAL_WARMSTART_RESULTS=str(tmp_path / 'root'),
               PILOT_PY=str(stub / 'python'), STUB_ARGS=str(args_file), RUN_NAME='regress_l2_d4_b64')
    r = subprocess.run(['bash', LAUNCH, 'regress', '1', '--prediction_type', 'direct', '--loss_type', 'l2'], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    run = tmp_path / 'root' / 'runs' / 'regress_l2_d4_b64'
    assert (run / 'launches').read_text().strip() == '1'
    assert not (tmp_path / 'root' / 'runs' / 'regress_d4_b64').exists()
    deadline = time.monotonic() + 10
    while not args_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    a = args_file.read_text()
    assert '--mode regress' in a and '--device cuda:1' in a and f'--save_dir {run}' in a
    assert a.rstrip().endswith('--prediction_type direct --loss_type l2')
```

- [ ] **Step 2: Run it to verify it fails**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_launch_run_name.py -q`
Expected: FAIL. `launches` is missing under `regress_l2_d4_b64` because the script wrote `regress_d4_b64`.

- [ ] **Step 3: Implement and write the smoke script**

`scripts/pilot/launch_run.sh`:
- `RUN=$ROOT/runs/${MODE}_d4_b64` becomes `RUN=$ROOT/runs/${RUN_NAME:-${MODE}_d4_b64}     # RUN_NAME overrides the run directory (e.g. regress_l2_d4_b64)`.
- The usage comment becomes `# Usage: [RUN_NAME=<dir name>] launch_run.sh residual|standard|standard_xhat|regress GPU [extra args]   (regress needs --prediction_type direct --loss_type l1|l2)`.

`scripts/pilot/smoke_regress.sh`:

```bash
#!/bin/bash
# GPU smoke test of mode regress on 64 training files: train 2 epochs, resume to 3, evaluate the direct grid
# (meta-checked, loss recorded), and confirm that evaluating the regressor as B is refused. Usage: smoke_regress.sh GPU
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python
ROOT=${RESIDUAL_WARMSTART_RESULTS:-/data/chaoyi_he/HSI/Diffu/results/residual_warmstart}
GPU=${1:-0}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
NAME=smoke_regress_gpu${GPU}_d4_b16
RUN=$ROOT/runs/$NAME
rm -rf "$RUN"
COMMON="main_2d_residual_fsdp.py --mode regress --prediction_type direct --loss_type l2 --ds 4 --base_channels 16 --batch_size 8 --num_workers 8 --device cuda:$GPU --limit_files 64 --save_every 1 --keep_epochs 1 2 3 --val_every 1 --save_dir $RUN"
$PY $COMMON --num_epochs 2
[[ "$(cat "$RUN/DONE")" == "2" ]] || { echo 'no DONE = 2 after 2 epochs'; exit 1; }
$PY $COMMON --num_epochs 3 --resume auto
[[ "$(cat "$RUN/DONE")" == "3" ]] || { echo 'resume did not reach epoch 3'; exit 1; }
$PY scripts/eval_warmstart.py --d 4 --ckpt "$RUN/checkpoint_epoch_3.pth" --mode regress --prediction_type direct \
    --grid direct --base_channels 16 --seeds 0 1 --batch_size 8 --device "cuda:$GPU"
JSON=$ROOT/eval/regress/d4/${NAME}__checkpoint_epoch_3__direct.json
[[ -f "$JSON" ]] || { echo "missing $JSON"; exit 1; }
grep -q '"meta_checked": true' "$JSON" || { echo 'meta not checked'; exit 1; }
grep -q '"loss_type": "l2"' "$JSON" || { echo 'loss_type not recorded'; exit 1; }
if ERR=$($PY scripts/eval_warmstart.py --d 4 --ckpt "$RUN/checkpoint_epoch_3.pth" --mode residual --grid full \
         --base_channels 16 --seeds 0 --device "cuda:$GPU" 2>&1); then
  echo 'evaluating the regressor as B was NOT refused'; exit 1
fi
grep -q 'does not match this evaluation' <<< "$ERR" || { echo "refused for another reason: $ERR"; exit 1; }
rm -rf "$RUN" "$JSON"
echo "regress smoke test (GPU $GPU) passed"
```

`chmod +x scripts/pilot/smoke_regress.sh`; `bash -n` both scripts.

- [ ] **Step 4: Run the tests, then the GPU smoke and the timing**

Run:
1. `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_launch_run_name.py tests/test_pilot_driver.py -q`, then the full suite. Expected: 247 passed.
2. `nvidia-smi --query-compute-apps=pid --format=csv,noheader`. Expected: empty (no compute process).
3. `bash scripts/pilot/smoke_regress.sh 0`. Expected: `regress smoke test (GPU 0) passed` in ≈ 6 min.
4. `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python main_2d_residual_fsdp.py --mode regress --prediction_type direct --loss_type l2 --ds 4 --base_channels 64 --batch_size 8 --num_workers 8 --device cuda:0 --save_dir /tmp/unused --dry_run_steps 200`. Expected: a JSON record with `sec_per_step ≈ 0.23`, `data_wait_sec` < 10 % of it, and finite loss.
5. Append the record to `RESULTS_ROOT/timing_phase1.json` (a JSON list) with a helper script, and confirm with `nvidia-smi` that no process is left.

- [ ] **Step 5: Commit**

```bash
git add scripts/pilot/smoke_regress.sh tests/test_launch_run_name.py
git commit -F <msgfile> -- scripts/pilot/launch_run.sh scripts/pilot/smoke_regress.sh tests/test_launch_run_name.py
```
Message: `feat(warmstart): launch_run.sh RUN_NAME; regress GPU smoke test (train, resume, direct-grid eval, meta refusal); regress timing record` + a blank line + the dry-run record + trailers.

---

### Task 6: Step-0 diagnostics run (GPU, ≈ 1 h; no code)

**Files:**
- Create (git-ignored), all under `RESULTS_ROOT/diagnostics/`:
  - `b_fast.npz`, `b_ddpm_a.npz`, `b_ddpm_b.npz`, `b_diagnostics.json`;
  - `yswap_residual_d4_b64_t999_n32.json`;
  - logs `*.log`.

This task is procedural. It records what it ran and what came out.

- [ ] **Step 1: Check the GPUs and prepare**

Run: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` and `nvidia-smi --query-compute-apps=pid --format=csv,noheader`. Both GPUs must be idle and the second command must print nothing. Then `mkdir -p /data/chaoyi_he/HSI/Diffu/results/residual_warmstart/diagnostics`.

- [ ] **Step 2: Launch the three sampling jobs**

Run each as one background command from the worktree, with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`. `DIAG` = `/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/diagnostics`.
- GPU 1:
  ```
  nohup python scripts/diagnose_b.py sample --rows ddpm1000 --draws 2 3 --device cuda:1 --out $DIAG/b_ddpm_b.npz > $DIAG/b_ddpm_b.log 2>&1 &
  ```
  ≈ 45 min.
- GPU 0, first:
  ```
  nohup python scripts/diagnose_b.py sample --rows ddim10 t999 --draws 0 1 2 3 4 5 6 7 --device cuda:0 --out $DIAG/b_fast.npz > $DIAG/b_fast.log 2>&1 &
  ```
  ≈ 8 min.
- GPU 0, when that finishes:
  ```
  nohup python scripts/diagnose_b.py sample --rows ddpm1000 --draws 0 1 --device cuda:0 --out $DIAG/b_ddpm_a.npz > $DIAG/b_ddpm_a.log 2>&1 &
  ```
  ≈ 45 min.

- [ ] **Step 3: B's one-step y-swap on 32 cubes (CPU, in parallel)**

Run:
```
CUDA_VISIBLE_DEVICES= python scripts/yswap_one_step.py --checkpoint /data/chaoyi_he/HSI/Diffu/results/residual_warmstart/runs/residual_d4_b64/checkpoint_epoch_100.pth --mode residual --t 999 --n 32 --out $DIAG/yswap_residual_d4_b64_t999_n32.json
```
Expected, as reproduced in the final review: true 2.2479, swap_full 2.4704, baseline 2.4194; swap_full − true +0.2225 ± 0.0136. If the numbers differ by more than 0.0005, report it.

- [ ] **Step 4: Merge**

When the three shards exist: `python scripts/diagnose_b.py merge --inputs $DIAG/b_fast.npz $DIAG/b_ddpm_a.npz $DIAG/b_ddpm_b.npz --out $DIAG/b_diagnostics.json` (CPU, ≈ 2 min).
Expected:
- no "WARNING: shards sampled on" line (all three shards are CUDA-sampled);
- `consistency_vs_stored` ≤ 1e-3 for draws 0 and 1 in every row of these CUDA-sampled shards (otherwise the seeding or batching differs from the harness: STOP and report);
- single-draw means close to the stored rows: t999 ≈ 2.2426, ddim10 ≈ 3.086, ddpm1000 ≈ 3.402;
- a `lambda_vs_one_step_centre` for ddim10 and ddpm1000, and a `lambda_vs_ddim1_single_draw` reference and a `calibration` block for every row.

Orientation, not a pass criterion: a CPU preview made while verifying this plan (t999 row only, K = 2, CPU noise, so independent of the stored draws) read:
- t999 single 2.2398 %, K-mean 2.2248 %;
- `ratio_of_means` 0.026 and per-cube spread slope 0.011 (reference 0.896): B's one-step output barely varies with its x_T draw;
- single-draw λ reference 1.28 (the bias the one-step-centre endpoint removes).

Record every row's summary line.

- [ ] **Step 5: Report**

No commit; the artefacts are git-ignored. Record in the task report:
- the commands, wall-clock times and GPU state before and after;
- the merged per-row table: K, n, single-draw RMSE, K-mean RMSE, pred H0, pred H1, λ ± jackknife SE against the one-step centre and the single-draw reference, calibration (`ratio_of_means`, `self_ratio`, per-cube slope ± SE against the reference slope), consistency;
- B's y-swap summary.

---

### Task 7: Train the L1 and L2 regressors, evaluate, gate; conditional second seed (GPU, ≈ 9 h + conditional ≈ 9 h)

**Files:**
- Create (git-ignored):
  - `RESULTS_ROOT/runs/regress_{l1,l2}_d4_b64/` and, conditionally, `regress_{l1,l2}_s43_d4_b64/`;
  - evaluations `RESULTS_ROOT/eval/regress/d4/*__direct.json`;
  - gate records `gate_1a_prime__*.json`.

- [ ] **Step 1: Launch both regressors**

Confirm both GPUs are idle (Task 6 finished). Then:
```
RUN_NAME=regress_l1_d4_b64 bash scripts/pilot/launch_run.sh regress 0 --prediction_type direct --loss_type l1
RUN_NAME=regress_l2_d4_b64 bash scripts/pilot/launch_run.sh regress 1 --prediction_type direct --loss_type l2
```
After ≈ 12 min run `python scripts/pilot/status.py`. Expected:
- both runs alive;
- ≈ 312–315 s/epoch side by side, `data` ≤ 10 %;
- finite losses, `skipped` 0.

Record each run's epoch-1 loss. If either run is slower than 380 s/epoch, report it with `uptime` and `top -b -n 1 | head -20`. Do not restart.

- [ ] **Step 2: Watch until both are DONE (≈ 8.8 h)**

Use a background watcher. It exits when both `DONE` files exist, or when a run's pid is dead and it has no `DONE`, and re-arms in ≤ 2 h stretches. A dead run is relaunched with the same command; `launch_run.sh` resumes it from the latest checkpoint and counts the launch. Record every relaunch.

- [ ] **Step 3: Evaluate both**

```
python scripts/eval_warmstart.py --d 4 --ckpt $RESULTS_ROOT/runs/regress_l1_d4_b64/checkpoint_epoch_100.pth --mode regress --prediction_type direct --grid direct --seeds 0 1 --device cuda:0
python scripts/eval_warmstart.py --d 4 --ckpt $RESULTS_ROOT/runs/regress_l2_d4_b64/checkpoint_epoch_100.pth --mode regress --prediction_type direct --grid direct --seeds 0 1 --device cuda:1
```
Expected: `eval/regress/d4/regress_{l1,l2}_d4_b64__checkpoint_epoch_100__direct.json`, each with `meta_checked` true and a 32-cube y-swap at t = 999. Each takes ≈ 2 min.

- [ ] **Step 4: Gate**

```
python scripts/gate_1a_prime.py --reg $RESULTS_ROOT/eval/regress/d4/regress_l1_d4_b64__checkpoint_epoch_100__direct.json $RESULTS_ROOT/eval/regress/d4/regress_l2_d4_b64__checkpoint_epoch_100__direct.json --b_yswap $RESULTS_ROOT/diagnostics/yswap_residual_d4_b64_t999_n32.json --b_diag $RESULTS_ROOT/diagnostics/b_diagnostics.json
```
Record the printed block verbatim, including `best`, `best_verdict` and each model's `target_met`.

- [ ] **Step 5: Conditional second seed**

Run this step only if `best_verdict` is `PASS` and the best model's `target_met` is true (`best` is the lowest-RMSE `PASS` regressor):
1. Launch seed 43 of both losses side by side (both GPUs idle):
   ```
   RUN_NAME=regress_l1_s43_d4_b64 bash scripts/pilot/launch_run.sh regress 0 --prediction_type direct --loss_type l1 --seed 43
   RUN_NAME=regress_l2_s43_d4_b64 bash scripts/pilot/launch_run.sh regress 1 --prediction_type direct --loss_type l2 --seed 43
   ```
   This takes ≈ 8.8 h, watched as in Step 2.
2. Evaluate both as in Step 3.
3. Re-run the gate on all four evaluations: `--reg <l1> <l2> <l1 s43> <l2 s43> --b_yswap ... --b_diag ...`. The record holds a loss pair per seed and a seed pair per loss.
4. Record both seed pairs and `go_1b`. `go_1b` is true when either loss's seed pair qualifies; record which loss.

Otherwise skip this step and record why: `best_verdict`, and `target_met`. If `best_verdict` is `PASS` but the target is not met, seed 43 cannot make `go_1b` true; run it only if the user asks for a seed-variance estimate.

---

### Task 8: Results note

**Files:**
- Create: `docs/superpowers/plans/2026-10-12-residual-warm-start-phase1a-prime-RESULTS.md`
- Modify: `docs/superpowers/plans/2026-10-09-residual-warm-start-phase1a-RESULTS.md`, Decision step 0:
  - "(minutes of GPU, no training)" becomes "(≈ 1.5 h of GPU, ≈ 55 min wall-clock over both GPUs: DDPM-1000 K = 4 on 50 cubes is 1.46 GPU-hours; see the Phase 1a′ note)".
  - After "(gap 0.088 ± 0.009)" insert "; the H1 value used the single-draw ddim1 energy, which includes the one-step output's own noise energy, and the Phase 1a′ note replaces it with the prediction from the noise-free t = 999 centre".

- [ ] **Step 1: Write the note** from the artefacts:
  - `b_diagnostics.json`;
  - `yswap_residual_d4_b64_t999_n32.json`;
  - each regressor run's `train_log.jsonl`, `args.json` and `launches`;
  - the regress evaluations;
  - the gate record(s);
  - `timing_phase1.json`;
  - the Phase 1a evaluations.

  Every number must trace to a file; fill every cell. Template:

```markdown
# Phase 1a′ results — <date>

**Verdicts:** <per regressor: verdict, RMSE, target met?>; B under the proposed rule: <verdict>; go_1b: <true|false>.

## Step 0 — what B's chain does (b_diagnostics.json)
| row | cubes | K | single-draw RMSE | K-mean RMSE | pred H0 | pred H1 | λ vs one-step centre (± jk SE) | λ vs single-draw ddim1 (biased reference) | calibration ratio / self ratio | spread slope on b² ± SE (reference slope) | draws 0/1 vs stored |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ddim10 | 200 | 8 | | | | | | | | | |
| t999 | 200 | 8 | | | | | 1 (by construction) | | | | |
| ddpm1000 | 50 | 4 | | | | | | | | | |
Reading:
- λ ≈ 1 means the chain is centred on the noise-free one-step centre (a conditional posterior sampler); λ ≈ 0 means it samples an unconditional residual.
- The spread is calibrated only if `ratio_of_means` ≈ 1 AND the per-cube slope of spread on b² is close to the reference slope of the one-step centre energy on b². A slope ≈ 0 means a cube-independent spread (Phase 1a Observation 1), whatever the ratio.

## Regressor runs (d = 4, base 64, batch 8, side by side on GPU 0 / GPU 1; 100 epochs)
| run | loss | seed | wall h | s/epoch median (data %) | loss epoch 1 → 100 | val loss (800 non-gate files) epoch 5 → 100 | peak GB | skipped | launches |
|---|---|---|---|---|---|---|---|---|---|

## Comparison on the 200 gate cubes (paired per cube; ± iid SE, clustered SE in brackets)
| model | NFE | RMSE % | − baseline | − B ddim1 | − B t999 8-draw mean (report only) | P022 | unseen | target 1.9985 % |
|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 2.3512 | — | — | — | | | |
| B ddim1 | 1 | 2.2426 | −0.1086 ± 0.0060 (0.0068) | — | | | | |
| B t999 mean of K = 8 draws | 8 | | | | — | | | |
| R L1 | 1 | | | | | | | |
| R L2 | 1 | | | | | | | |
| R L1 seed 43 (if run) | 1 | | | | | | | |
| R L2 seed 43 (if run) | 1 | | | | | | | |
L1 − L2 (paired, per seed): <diff ± SE>. Seed pairs (if run): <diff ± SE per loss>, go_1b: <...>.

## Validity — proposed one-step y-swap rule (t = 999, 32 cubes)
| model | true | swap_full | baseline | swap_full − true | swap_full − baseline | valid (invalid_reason) | spec 0.9 ratio |
|---|---|---|---|---|---|---|---|

## Gate 1a′
<verbatim gate_1a_prime.py output>

## Observations
- What step 0 says about B's chain and the √2 question.
- Regressor vs B: what direct regression at t = 999 changes (the objective, training at a single t, a noise-free input), judged against both B's single draw and its 8-draw mean. The objective is credited only if `vs_b_t999_kmean` also passes. A regressor that beats B's single draw but not its 8-draw mean shows B's noise sensitivity, not a better objective.
- L1 vs L2 under RMSE.
- Seen vs unseen classes.
- Compute: wall-clock per run, deviations, restarts.

## Decision for the user
- go_1b true → propose Phase 1b′: the regressor at base 256, FSDP, batch 8/GPU (27.6 h estimate from Phase 0 timing), 200 epochs, same gate.
- Seed 42 PASS with the target met, but seed 43 not PASS or not meeting the target → report the seed-pair difference ± SE; no 1b: the margin is within training-seed variance.
- Best regressor PASS but target not met → <options: more capacity/epochs, loss weighting, both>.
- PASS_NOT_OVER_B / FAIL → at base 64, 100 epochs and one seed this is a lower bound on what (x̂₀, y_d) predict at d = 4. Separate information from capacity/optimisation (val-loss trajectory, L1 vs L2) before concluding; then the options left (conditioning path, data, d).
- INVALID (a) (the swap changes nothing) → the regressor ignores its condition; diagnose before any further run.
- INVALID (b) only (swap_full below the baseline) → part of the gain is cube-independent; report the split baseline − swap_full vs swap_full − true; no 1b until resolved.
```

- [ ] **Step 2: Verify every number**

Write a helper that re-derives every number in the note from its sources and reports 0 mismatches. Run the full test suite once and put its summary line in the note's compute bullet.

- [ ] **Step 3: Commit**

```bash
git commit -F <msgfile> -- docs/superpowers/plans/2026-10-12-residual-warm-start-phase1a-prime-RESULTS.md docs/superpowers/plans/2026-10-09-residual-warm-start-phase1a-RESULTS.md
```
(Run `git add` on the new note first.) Message: `docs(warmstart): Phase 1a-prime results — B's chain diagnostics, one-step residual regressors (L1/L2) and gate` + trailers.

Then present the note to the user. This is the next decision point: whether to go to a full-size run.
