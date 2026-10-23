"""Residual warm-start diffusion training (spec 2026-10-08, Phase 1).

Single GPU:  python main_2d_residual_fsdp.py --mode residual --ds 4 --base_channels 64 --batch_size 8 --device cuda:0 \
                 --save_dir /data/chaoyi_he/HSI/Diffu/results/residual_warmstart/runs/residual_d4_b64
FSDP:        torchrun --standalone --nproc_per_node=2 main_2d_residual_fsdp.py --mode residual --ds 4 --base_channels 256 \
                 --batch_size 4 --num_epochs 200 --max_hours 0 --save_dir .../runs/residual_d4_b256
Timing only: add --dry_run_steps 200 (sec/step with and without data wait, peak memory; writes nothing).

Wall clock: --max_hours 0 (the default) means no cap. With a cap > 0 the run stops after the epoch that crosses it,
saves that epoch's checkpoint, marks its train_log record stopped_on_wallclock, and writes STOPPED (JSON: epoch, reason,
max_hours, hours, num_epochs) next to DONE. DONE always holds the bare final epoch, so "DONE without STOPPED" is a
finished schedule. Every launch first removes a stale DONE / STOPPED, and refuses (SystemExit) to resume a checkpoint
that already used the whole --max_hours budget.

Modes (model/residual_wrapper.py): residual (B), standard (A), standard_xhat (A'). --batch_size is PER GPU.
Mode regress (R, the one-step residual regressor): --mode regress --prediction_type direct --loss_type l1|l2.
Conventions taken from main_2d_fsdp.py (FSDP wrap, bf16, AdamW + cosine per epoch, FULL_STATE_DICT checkpoints) without
its rank-0-only checkpoint calls, worker cap and save-path rewriting. Every distributed decision (non-finite skip, memory
refusal, wall-clock stop) is collective. No ds warm-up: d is fixed because x0_hat and sigma_d depend on it.

CPU threads: each DataLoader worker computes x0_hat with numpy BLAS and is limited to one BLAS/OpenMP thread by its
worker_init_fn (data_loader.hfd_residual.limit_worker_threads). That limit is what protects single-GPU and cluster
launches: torchrun sets OMP_NUM_THREADS=1 only when it is unset and nproc_per_node > 1, and the thread exports in
scripts/pilot/launch_run.sh cover only the pilot. The main process keeps its default threads.
"""
import argparse
import datetime
import functools
import json
import math
import os
import random
import re
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
from data_loader.hfd_residual import HFDResidualData, limit_worker_threads, residual_collate_fn  # noqa: E402
from data_loader.warmstart_paths import DATA_ROOT, PRIOR, RESULTS_ROOT, SPLIT_JSON, SUBSET, file_sha256, git_head  # noqa: E402,F401
from data_loader.xr_cache import XRCachedData  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from model.regression_trainer import RegressionTrainer  # noqa: E402
from model.residual_wrapper import MODES, XR_MODES, build_residual_model, target_for  # noqa: E402
from model.varhead_trainer import VarheadTrainer  # noqa: E402
from scripts.uq.split_halves import read_split  # noqa: E402

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
    ap.add_argument('--max_hours', type=float, default=0.0,
                    help='stop after the epoch that crosses this wall-clock budget, counted over all launches (0 = no cap)')
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--lrf', type=float, default=0.033, help='eta_min = lr * lrf')
    ap.add_argument('--weight_decay', type=float, default=0.0); ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--prediction_type', choices=['eps', 'x0', 'v', 'direct'], default='v',
                    help="diffusion parameterisation; 'direct' is required by, and only allowed with, --mode regress / varhead")
    ap.add_argument('--loss_type', choices=['l1', 'l2', 'nll'], default='l1')
    ap.add_argument('--train_half', type=int, choices=[0, 1, 2], default=0,
                    help='Phase 2a: train a regressor on half 1 or 2 of the cross-fitting split only (0 = all files)')
    ap.add_argument('--split_json', default=SPLIT_JSON, help='the cross-fitting split (scripts/uq/split_halves.py)')
    ap.add_argument('--xr_cache', default='', help='Phase 2a: regressor-centre cache dir (required by --mode residual_reg / varhead)')
    ap.add_argument('--snr_gamma', type=float, default=5.0, help='min-SNR-gamma loss weighting, <= 0 = off (ignored by --mode regress / varhead)')
    ap.add_argument('--val_every', type=int, default=5, help='loss-only validation on the non-gate validation files every N epochs (0 = never)')
    ap.add_argument('--save_every', type=int, default=10); ap.add_argument('--keep_last', type=int, default=2)
    ap.add_argument('--keep_epochs', type=int, nargs='*', default=[25, 50, 100], help='milestone checkpoints exempt from pruning')
    ap.add_argument('--max_nonfinite', type=int, default=10, help='abort after this many consecutive skipped steps (non-finite loss or gradient norm)')
    ap.add_argument('--log_interval', type=int, default=200)
    ap.add_argument('--save_dir', required=True); ap.add_argument('--resume', default='', help='checkpoint path; "auto" = latest in --save_dir')
    ap.add_argument('--device', default='cuda:0', help='single-process device; ignored under torchrun')
    ap.add_argument('--no_bf16', action='store_true'); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sharding_strategy', choices=['FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD'], default='FULL_SHARD')
    ap.add_argument('--wrap_min_params', type=int, default=1_000_000)
    ap.add_argument('--dry_run_steps', type=int, default=0, help='time N steps (after warm-up) and exit without saving')
    return ap


# ----------------------------------------------------------------------------- setup
NCCL_TIMEOUT = datetime.timedelta(minutes=60)     # a full-size FULL_STATE_DICT gather + rank-0 save must fit inside it


def setup_dist(args):
    """torchrun sets WORLD_SIZE; without it this is a single-process run on args.device."""
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        dist.init_process_group('nccl', timeout=NCCL_TIMEOUT)
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
    n_before = len(val.img_list)
    val.img_list = [f for f in val.img_list if os.path.normpath(f) not in gate]       # monitoring never sees the gate cubes
    assert n_before - len(val.img_list) == len(gate), \
        f'only {n_before - len(val.img_list)} of {len(gate)} gate files were found in the validation split ' \
        f'(path-form mismatch?); refusing to monitor on gate cubes'
    # getattr: scripts/regress_monitoring_energy.py and scripts/diagnose_spatial_split.py pass only data_root, d, prior, limit_files
    train_half, xr_cache, split_json = getattr(args, 'train_half', 0), getattr(args, 'xr_cache', ''), getattr(args, 'split_json', SPLIT_JSON)
    if train_half:
        half = set(read_split(split_json)[f'half{train_half}'])
        n_before = len(train.img_list)
        train.img_list = [f for f in train.img_list if os.path.relpath(f, args.data_root) in half]
        if len(train.img_list) != len(half):
            raise ValueError(f'--train_half {train_half}: {len(train.img_list)} of the split\'s {len(half)} files are in the '
                             f'{n_before}-file training list (split {split_json} vs data_root {args.data_root})')
    if xr_cache:
        train = XRCachedData(train, xr_cache, 'train', args.data_root)
        val = XRCachedData(val, xr_cache, 'val', args.data_root)
    if args.limit_files:
        train.img_list = train.img_list[:args.limit_files]; val.img_list = val.img_list[:max(8, args.limit_files // 8)]
    if xr_cache:
        train.check_files(); val.check_files()
    return train, val


def make_loader(ds, args, batch_size, shuffle, rank, world, device):
    """Shuffled loaders always use a DistributedSampler (num_replicas=1 on one GPU) so the order of epoch e is a pure
    function of (seed, e) via set_epoch — the basis of exact resume. Every worker runs with one BLAS thread
    (limit_worker_threads); see the module docstring."""
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=shuffle, seed=args.seed, drop_last=shuffle) \
        if (shuffle or world > 1) else None
    kw = dict(num_workers=args.num_workers, pin_memory=(device.type == 'cuda'), collate_fn=residual_collate_fn,
              worker_init_fn=limit_worker_threads)
    if args.num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)         # safe: d never changes during a run
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=shuffle, **kw)


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
def half_split_sha256(args):
    """sha256 of the --split_json that defines a half regressor's half; None for --train_half 0 (or no such field)."""
    return file_sha256(getattr(args, 'split_json', SPLIT_JSON)) if getattr(args, 'train_half', 0) else None


def checkpoint_meta(args, sigma_d, prior_s, prior_sha256, extra=None):
    """What scripts/eval_warmstart.py::check_meta verifies (mode, d, base_channels, prediction_type, sigma_d, prior_sha256)
    plus provenance: n_timesteps, prior_s, the arguments, the checkout's git HEAD (None without git; uncommitted changes
    are not recorded) and the torch version. git_head and torch are never compared on resume or evaluation.
    Phase 2a: always train_half, and for a half regressor split_sha256 of its --split_json; for the XR modes `extra`
    (XRCachedData.checkpoint_extra()) adds sigma_r, xr_cache_sha256, centre and the cache's split_sha256, and sigma_d is
    sigma_r (decision D1). Raises ValueError rather than let the two split_sha256 overwrite each other."""
    meta = dict(mode=args.mode, d=args.d, base_channels=args.base_channels, prediction_type=args.prediction_type,
                n_timesteps=N_TIMESTEPS, sigma_d=float(sigma_d), prior_s=float(prior_s), prior_sha256=prior_sha256, args=vars(args),
                git_head=git_head(), torch=torch.__version__, train_half=int(getattr(args, 'train_half', 0)))
    split_sha = half_split_sha256(args)
    if split_sha is not None:
        meta['split_sha256'] = split_sha   # which split defined this half regressor's half
    if extra:
        if split_sha is not None and extra.get('split_sha256', split_sha) != split_sha:
            raise ValueError(f'split_sha256: --split_json {getattr(args, "split_json", SPLIT_JSON)} has {split_sha!r} '
                             f'but the dataset extra has {extra["split_sha256"]!r}')
        meta.update(extra)                 # XR modes: sigma_r, xr_cache_sha256, centre, split_sha256 (XRCachedData.checkpoint_extra)
    return meta


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


RESUME_META_KEYS = ('mode', 'd', 'base_channels', 'prediction_type')


def check_resume_meta(ck, args, sigma_d, prior_sha256, path='<checkpoint>', warn=True, cache_sha=None):
    """Refuse to resume a checkpoint whose `meta` differs from the current run (the provenance chain the gate relies on
    must not break silently: A' and B have identical parameter shapes, so strict loading would not notice a mode
    switch). Compares RESUME_META_KEYS and meta['args'] loss_type, seed and train_half exactly, sigma_d (|diff| < 1e-9),
    prior_sha256, when cache_sha is given (the XR modes) xr_cache_sha256 and, for a half regressor whose checkpoint
    names its split, split_sha256 against the current --split_json; raises ValueError naming every mismatch,
    and also when the checkpoint has no meta. A scheduler T_max different from --num_epochs only warns (the cosine
    schedule keeps the checkpoint's T_max; the Task 3 smoke test deliberately resumes 2 -> 3 epochs). Returns True."""
    meta = ck.get('meta') if isinstance(ck, dict) else None
    if not meta:
        raise ValueError(f'checkpoint {path} carries no meta; refusing to resume (every checkpoint this script writes has one)')
    bad = [f'{k}: checkpoint {meta.get(k, "<missing>")!r} vs args {getattr(args, k)!r}'
           for k in RESUME_META_KEYS if k not in meta or meta[k] != getattr(args, k)]
    margs = meta.get('args') or {}
    for k in ('loss_type', 'seed', 'train_half'):  # meta['args'] is vars(args) at save time; the L1/L2 runs differ only in loss_type
        if k in margs and margs[k] != getattr(args, k, None):
            bad.append(f'{k}: checkpoint {margs[k]!r} vs args {getattr(args, k, None)!r}')
    if 'sigma_d' not in meta or abs(float(meta['sigma_d']) - sigma_d) >= 1e-9:
        bad.append(f'sigma_d: checkpoint {meta.get("sigma_d", "<missing>")!r} vs prior {sigma_d!r}')
    if meta.get('prior_sha256') != prior_sha256:
        bad.append(f'prior_sha256: checkpoint {meta.get("prior_sha256", "<missing>")!r} vs prior file {prior_sha256!r}')
    if cache_sha is not None and meta.get('xr_cache_sha256') != cache_sha:
        bad.append(f'xr_cache_sha256: checkpoint {meta.get("xr_cache_sha256", "<missing>")!r} vs cache {cache_sha!r} (the cache was rebuilt)')
    split_sha = half_split_sha256(args)
    if split_sha is not None and 'split_sha256' in meta and meta['split_sha256'] != split_sha:
        bad.append(f'split_sha256: checkpoint {meta["split_sha256"]!r} vs split file {split_sha!r} '
                   f'({getattr(args, "split_json", SPLIT_JSON)})')
    if bad:
        raise ValueError(f'checkpoint {path} does not match the current arguments: ' + '; '.join(bad))
    t_max = ck.get('scheduler_state_dict', {}).get('T_max')
    if warn and t_max != args.num_epochs:
        print(f'WARNING: checkpoint scheduler T_max={t_max} != --num_epochs {args.num_epochs}; '
              f'the cosine schedule keeps the checkpoint\'s T_max', flush=True)
    return True


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


def write_atomic(path, text):
    """Write a small marker/log file via a temp file + os.replace: a reader never sees it half-written."""
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w') as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def trim_train_log(path, last_epoch):
    """Keep only the records of epochs <= last_epoch, the ones the restored weights went through, so the epochs a resume
    retrains are not logged twice. A restart from scratch (last_epoch 0) empties it. Lines that do not parse (a
    half-written last line) are dropped. Rewritten atomically, only when something is dropped; returns how many lines."""
    if not os.path.exists(path):
        return 0
    keep, dropped = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ok = int(json.loads(line)['epoch']) <= last_epoch
            except (ValueError, KeyError, TypeError):
                ok = False
            if ok:
                keep.append(line if line.endswith('\n') else line + '\n')
            else:
                dropped += 1
    if dropped:
        write_atomic(path, ''.join(keep))
    return dropped


def check_wallclock_budget(ck, args, path='<checkpoint>'):
    """Refuse to resume a checkpoint that already used the whole --max_hours budget (it would train one more epoch past
    the cap). --max_hours 0 means no cap."""
    used = float(ck.get('elapsed_hours', 0.0))
    if args.max_hours > 0 and used >= args.max_hours:
        raise SystemExit(f'checkpoint {path} has already used {used:.3f} h of the --max_hours {args.max_hours:g} budget; '
                         f'refusing to train past it (raise --max_hours, or pass --max_hours 0 for no cap)')


# ----------------------------------------------------------------------------- loops
def clip_grads(model, clip, dist_ctx):
    """Clip to `clip` (no clipping when clip <= 0) and return the total gradient norm before clipping. Under FSDP the norm
    is all-reduced over the shards, so every rank gets the same value and makes the same skip decision."""
    max_norm = clip if clip and clip > 0 else float('inf')                # inf: measure only (the factor clamps to 1)
    return float(model.clip_grad_norm_(max_norm)) if dist_ctx.enabled else float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))


def train_one_epoch(model, trainer, loader, optimizer, args, device, dist_ctx, mode, sigma_d, log_prefix=''):
    """One epoch. A step whose loss or total gradient norm is non-finite is skipped on every rank (no optimizer step);
    `skipped_nonfinite` counts both kinds (`skipped_nonfinite_grad` the second), and max_nonfinite consecutive skips
    abort. grad_norm_mean / grad_norm_max are over the steps taken (pre-clip norms; None when no step was taken)."""
    model.train()
    use_bf16 = not args.no_bf16 and device.type == 'cuda'
    tot, n, skipped, skipped_grad, run_skipped, t_data, t0 = 0.0, 0, 0, 0, 0, 0.0, time.time()
    norms = []
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
                raise RuntimeError(f'{run_skipped} consecutive non-finite steps (loss or gradient norm) at step {step}; aborting')
            t_fetch = time.time(); continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = clip_grads(model, args.grad_clip, dist_ctx)
        if not math.isfinite(gn):                                     # the all-reduced norm: every rank skips together
            skipped += 1; skipped_grad += 1; run_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            if run_skipped >= args.max_nonfinite:
                raise RuntimeError(f'{run_skipped} consecutive non-finite steps (loss or gradient norm) at step {step}; aborting')
            t_fetch = time.time(); continue
        run_skipped = 0
        optimizer.step()
        norms.append(gn)
        tot += loss.item(); n += 1
        if dist_ctx.rank == 0 and args.log_interval and step % args.log_interval == 0:
            print(f'{log_prefix}step {step}/{len(loader)} loss {loss.item():.4f} lr {optimizer.param_groups[0]["lr"]:.2e}', flush=True)
        t_fetch = time.time()
    agg = torch.tensor([tot, float(n)], device=device)
    if dist_ctx.enabled:
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
    return dict(loss=(agg[0] / max(agg[1], 1)).item(), lr=optimizer.param_groups[0]['lr'], steps=n,
                skipped_nonfinite=skipped, skipped_nonfinite_grad=skipped_grad,
                grad_norm_mean=float(np.mean(norms)) if norms else None, grad_norm_max=float(np.max(norms)) if norms else None,
                sec=time.time() - t0, data_sec=t_data)


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
    check_mode_args(args)
    train_ds, val_ds = make_datasets(args)
    model = build_residual_model(args.mode, args.base_channels)
    n_params = sum(p.numel() for p in model.parameters())           # counted before the wrap (sharding hides it after)
    resume_path = latest_checkpoint(args.save_dir) if args.resume == 'auto' else (args.resume or None)
    ck = load_checkpoint(resume_path) if resume_path else None
    if ck is not None:
        cache_sha = train_ds.checkpoint_extra()['xr_cache_sha256'] if args.xr_cache else None
        check_resume_meta(ck, args, train_ds.sigma_d, file_sha256(args.prior), path=resume_path, warn=dist_ctx.rank == 0, cache_sha=cache_sha)
        check_wallclock_budget(ck, args, resume_path)                    # before DONE / STOPPED are touched
        model.load_state_dict(ck['model_state_dict'], strict=True)          # before the wrap, on CPU
    model = wrap_fsdp(model, args, dist_ctx) if dist_ctx.enabled else model.to(device)
    trainer = make_trainer(args, device)
    optimizer = make_optimizer(model, args); scheduler = make_scheduler(optimizer, args)
    start_epoch, elapsed = (0, 0.0) if ck is None else restore_training_state(ck, model, optimizer, scheduler, dist_ctx)
    return SimpleNamespace(train_ds=train_ds, val_ds=val_ds, model=model, trainer=trainer, optimizer=optimizer, scheduler=scheduler,
                           start_epoch=start_epoch, elapsed=elapsed, resume_path=resume_path, n_params=n_params)


def run_epochs(args, parts, dist_ctx, device):
    """The epoch loop. `parts` is either build_everything()'s namespace or the tests' tuple (ds, loader, model, trainer, opt, sched).
    Before the first epoch rank 0 removes DONE / STOPPED and cuts train_log.jsonl back to the epochs <= start_epoch."""
    if isinstance(parts, tuple):
        ds, train_loader, model, trainer, optimizer, scheduler = parts
        val_loader, start_epoch, elapsed, sigma_d = None, 0, 0.0, ds.sigma_d
    else:
        model, trainer, optimizer, scheduler = parts.model, parts.trainer, parts.optimizer, parts.scheduler
        start_epoch, elapsed, sigma_d = parts.start_epoch, parts.elapsed, parts.train_ds.sigma_d
        train_loader = make_loader(parts.train_ds, args, args.batch_size, True, dist_ctx.rank, dist_ctx.world, device)
        val_loader = make_loader(parts.val_ds, args, args.batch_size, False, dist_ctx.rank, dist_ctx.world, device)
    ds_meta = parts[0] if isinstance(parts, tuple) else parts.train_ds
    extra = ds_meta.checkpoint_extra() if hasattr(ds_meta, 'checkpoint_extra') else None
    meta = checkpoint_meta(args, sigma_d, ds_meta.prior['s'], file_sha256(args.prior) if os.path.exists(args.prior) else '', extra=extra)
    t_start = time.time() - elapsed * 3600                           # before any rank-0-only work: the ranks' clocks agree
    done_path, stopped_path = os.path.join(args.save_dir, 'DONE'), os.path.join(args.save_dir, 'STOPPED')
    if dist_ctx.rank == 0:                                           # a live run never shows a stale DONE / STOPPED
        for p in (done_path, stopped_path):
            if os.path.exists(p):
                os.remove(p)
        dropped = trim_train_log(os.path.join(args.save_dir, 'train_log.jsonl'), start_epoch)
        if dropped:
            print(f'train_log.jsonl: dropped {dropped} record(s) after epoch {start_epoch} (retrained from the checkpoint)', flush=True)
    if dist_ctx.enabled:
        dist.barrier()
    final_epoch, stopped = start_epoch, None
    for epoch in range(start_epoch, args.num_epochs):
        epoch_seed(args, epoch, dist_ctx.rank)
        train_loader.sampler.set_epoch(epoch)
        rec = train_one_epoch(model, trainer, train_loader, optimizer, args, device, dist_ctx, args.mode, sigma_d, log_prefix=f'[epoch {epoch + 1}] ')
        scheduler.step()
        epoch1 = epoch + 1
        if val_loader is not None and args.val_every and epoch1 % args.val_every == 0:
            rec.update(validate(model, trainer, val_loader, args, device, dist_ctx, args.mode, sigma_d))
        hours = (time.time() - t_start) / 3600
        stop = torch.tensor(float(args.max_hours > 0 and hours >= args.max_hours), device=device)
        if dist_ctx.enabled:
            dist.broadcast(stop, src=0)                               # one decision for every rank
        out_of_time = bool(stop.item())
        rec.update(epoch=epoch1, elapsed_hours=hours, peak_gb=torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else 0.0)
        if out_of_time:
            rec['stopped_on_wallclock'] = True
            stopped = dict(epoch=epoch1, reason='max_hours', max_hours=args.max_hours, hours=hours, num_epochs=args.num_epochs)
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
                print(f'wall-clock budget {args.max_hours} h reached after epoch {epoch1} of {args.num_epochs}; stopping', flush=True)
            break
    if dist_ctx.rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        if stopped is not None:                                       # before DONE: whoever sees DONE also sees STOPPED
            write_atomic(stopped_path, json.dumps(stopped) + '\n')
        write_atomic(done_path, f'{final_epoch}\n')                  # bare integer, as read_int / check_done expect
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
        n_launch = 1 + max([int(m.group(1)) for f in os.listdir(args.save_dir) for m in [re.fullmatch(r'args_launch(\d+)\.json', f)] if m],
                           default=0)
        for name in (f'args_launch{n_launch}.json', 'args.json'):      # one record per launch; args.json = the latest
            with open(os.path.join(args.save_dir, name), 'w') as f:
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
