"""
2D U-Net Diffusion Model — FSDP multi-GPU training.

FSDP shards parameters, gradients, and optimizer states across all GPUs, giving
~N× lower per-GPU memory than DDP.  The 194 M-param model with AdamW needs
~4.5 GB just for optimizer states; FSDP divides that by world_size.

Launch (single node, 4 GPUs):
    torchrun --standalone --nproc_per_node=4 main_2d_fsdp.py [args]

Launch (multi-node SLURM, see job_fsdp.slurm):
    torchrun --nproc_per_node=$GPUS_PER_NODE \\
             --nnodes=$SLURM_NNODES \\
             --node_rank=$SLURM_NODEID \\
             --master_addr=$MASTER_ADDR \\
             --master_port=29500 \\
             main_2d_fsdp.py [args]

Differences from main_2d.py / main_2d_multi_gpu.py:
  - FSDP instead of DDP → parameters/grads/optim states sharded across GPUs
  - bf16 mixed precision via FSDP MixedPrecision (no GradScaler needed)
  - Checkpoint save uses FULL_STATE_DICT to consolidate shards on rank-0
  - Checkpoint load happens on CPU *before* FSDP wrapping
  - Gradient clipping uses model.clip_grad_norm_() (FSDP-aware global norm)
  - With bf16, the diffusion loss is accumulated in fp32 for stable backward
  - DistributedSampler with set_epoch() called each epoch
"""

import os
import json
import random
import time
import functools
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.u2net_hyperspectral import U2NetHyperspectral, U2NetBlock2D
from model.layers import BasicTransformerBlock
from model.diffusion_trainer import DiffusionTrainer
from data_loader.my_dataset import HASCID_data, image_collate_fn, pixel_collate_fn
from data_loader.HFD_dataset import HFD_data
from misc.util import MetricLogger, SmoothedValue


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='2D U-Net Diffusion Model — FSDP multi-GPU training'
    )

    # dataset
    parser.add_argument('--data_path', type=str,
                        default='dataset/HFD100 Mat dataset',
                        choices=['dataset/HASCID-Dataset', 'dataset/HFD100 Mat dataset'])
    parser.add_argument('--train_mode', type=str, default='image',
                        choices=['pixel', 'image'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Global batch size across all GPUs')
    parser.add_argument('--eval_ratio', type=float, default=0.1)
    parser.add_argument('--R-n', type=int, default=1, dest='R_n')
    parser.add_argument('--dataset', type=str, choices=['HASCID', 'HFD'], default='HFD')
    parser.add_argument('--ds', '--sensor_down_sample_rate', type=int, default=2,
                        dest='sensor_down_sample_rate',
                        help='Target downsample rate for the sensor condition. '
                             'During the first --ds_warmup_epochs epochs the rate '
                             'is held at 1 regardless of this value.')
    parser.add_argument('--ds_warmup_epochs', type=int, default=20,
                        help='Keep sensor_down_sample_rate=1 for this many epochs '
                             'before switching to --ds. Set 0 to disable warmup.')

    # model
    parser.add_argument('--spectral_channels', type=int, default=64)
    parser.add_argument('--sensor_channels', type=int, default=30)
    parser.add_argument('--base_channels', type=int, default=512)
    parser.add_argument('--use_channel_3d_conv', action='store_true',
                        help='Use channel-spatial Conv3d blocks in U-Net conv stages')
    parser.add_argument('--channel_kernel', type=int, default=7,
                        help='Kernel size along channel axis for channel-spatial Conv3d')
    parser.add_argument('--spatial_kernel', type=int, default=3,
                        help='Kernel size along H/W axes for channel-spatial Conv3d')
    parser.add_argument('--channel_num_filters', type=int, default=4,
                        help='Parallel 3D filters per ChannelSpatialConv block '
                             '(was 1 previously; >1 multiplies spectral expressivity)')

    # optimisation
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lrf', type=float, default=0.033,
                        help='LR decay factor (eta_min = lr * lrf). '
                             'Default 0.033 keeps eta_min at ~1e-6 given lr=3e-5 '
                             'to avoid bf16 precision collapse at tail of cosine.')
    parser.add_argument(
        '--grad_clip', type=float, default=1.0,
        help='Global L2 grad clip (FSDP-aware). Use 0 to disable. '
             'Values ~0.05 are very aggressive and often hurt diffusion training.')

    # training schedule
    parser.add_argument('--val_every', type=int, default=100,
                        help='Validate every N epochs (0 = never)')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=200)

    # diffusion
    parser.add_argument('--loss_type', type=str, default='l1',
                        choices=['l1', 'l2', 'huber'])
    parser.add_argument('--noise_schedule', type=str, default='linear',
                        choices=['linear', 'cosine'])
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--prediction_type', type=str, default='v',
                        choices=['eps', 'x0', 'v'],
                        help='v-prediction mixes x0/eps signal — much more stable '
                             'gradient at both high and low SNR than pure eps.')

    # FSDP-specific
    parser.add_argument('--sharding_strategy', type=str,
                        default='FULL_SHARD',
                        choices=['FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD'],
                        help='FULL_SHARD=ZeRO-3 (max savings), '
                             'SHARD_GRAD_OP=ZeRO-2, NO_SHARD=DDP-equivalent')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='Disable bf16 mixed precision (use fp32 throughout)')
    parser.add_argument('--cpu_offload', action='store_true',
                        help='Offload parameters and gradients to CPU (very slow, '
                             'use only when GPU memory is extremely limited)')
    parser.add_argument('--wrap_min_params', type=int, default=1_000_000,
                        help='Wrap sub-modules with >= this many parameters '
                             'as separate FSDP units')

    # output / resume
    parser.add_argument('--resume', type=str,
                        default='results/2d_hsi_diffusion/down_sample_2/HFD/R_1/l1_loss_fsdp_3dconv/checkpoint_epoch',
                        help='Path to checkpoint (loaded on CPU before FSDP wrapping)')
    parser.add_argument('--save_path', type=str,
                        default='results/2d_hsi_diffusion/down_sample_2/HFD/R_1/l1_loss_fsdp')
    parser.add_argument('--seed', type=int, default=42)

    return parser


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialise NCCL process group using torchrun env-vars."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# FSDP checkpoint utilities
# ---------------------------------------------------------------------------

def save_fsdp_checkpoint(model, optimizer, scheduler, epoch, loss, save_path):
    """
    Gather full (un-sharded) state dicts on rank-0 and save to disk.

    Both model and optimizer state dicts are consolidated on rank-0 only;
    other ranks return immediately after the collective communication.
    """
    rank = dist.get_rank()

    # --- model state dict ---
    model_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, model_cfg):
        model_state = model.state_dict()

    # --- optimizer state dict ---
    optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                               optim_state_dict_config=optim_cfg):
        optim_state = FSDP.optim_state_dict(model, optimizer)

    # --- only rank-0 writes ---
    if rank == 0:
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model_state,
            'optimizer_state_dict': optim_state,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f'[rank 0] Checkpoint saved → {save_path}')

    dist.barrier()


def load_checkpoint_to_cpu(resume_path):
    """
    Load a checkpoint on CPU.  Must be called BEFORE FSDP wrapping so that the
    full state dict can be broadcast via load_state_dict().
    """
    if resume_path and resume_path.endswith('.pth') and os.path.exists(resume_path):
        print_rank0(f'Loading checkpoint from {resume_path} …')
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        return ckpt
    return None


# ---------------------------------------------------------------------------
# FSDP model factory
# ---------------------------------------------------------------------------

def wrap_model_with_fsdp(model, args, local_rank):
    """
    Wrap the plain nn.Module with FSDP.

    Wrapping strategy
    -----------------
    We use size_based_auto_wrap_policy so every sub-module with at least
    `args.wrap_min_params` parameters becomes a separate FSDP unit.
    With base_channels=256, each U2NetBlock2D has ~10-50 M parameters,
    so the default threshold of 1 M wraps each block independently.
    This gives fine-grained sharding with low communication overhead.
    """
    strategy_map = {
        'FULL_SHARD':    ShardingStrategy.FULL_SHARD,
        'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
        'NO_SHARD':      ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = strategy_map[args.sharding_strategy]

    # bf16 mixed precision: params+buffers in bf16, gradient reduction in fp32
    if not args.no_mixed_precision:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,   # fp32 reduction for numerical stability
            buffer_dtype=torch.bfloat16,
        )
    else:
        mixed_precision = None

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=args.wrap_min_params,
    )

    cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # overlap comms with compute
        cpu_offload=cpu_offload,
        device_id=local_rank,
        sync_module_states=True,   # broadcast rank-0 params to all ranks after wrapping
        use_orig_params=True,      # required for per-param grad clipping / LR tricks
    )
    return model


# ---------------------------------------------------------------------------
# Training loop (FSDP-aware)
# ---------------------------------------------------------------------------

def train_one_epoch(model, diffusion_trainer, loader, optimizer, epoch,
                    local_rank, world_size, grad_clip, log_interval,
                    use_bf16=True, verbose_log=True):
    """
    One training epoch.  Key FSDP differences from the single-GPU loop:
      - model.clip_grad_norm_() computes a true global gradient norm across shards
      - No GradScaler — bf16 doesn't require loss scaling
      - autocast dtype matches FSDP MixedPrecision (bfloat16)
      - verbose_log=False on non-main ranks avoids duplicate tqdm-style lines
    """
    model.train()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch + 1}]'

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    device = torch.device(f'cuda:{local_rank}')

    for step, batch in enumerate(
        metric_logger.log_every(loader, log_interval, header, verbose=verbose_log)
    ):
        data, conditions = batch
        data       = data.to(device, non_blocking=True)
        conditions = conditions.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_bf16):
            loss, _ = diffusion_trainer.get_loss(
                model, data, conditions, loss_in_fp32=use_bf16
            )

        loss.backward()

        # FSDP-aware gradient clipping — computes global norm across all shards
        if grad_clip is not None and grad_clip > 0:
            model.clip_grad_norm_(grad_clip)

        optimizer.step()

        batch_size = data.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['loss'].update(loss.item(), n=batch_size)

    # Synchronise epoch-average loss across all ranks
    avg_loss_tensor = torch.tensor(
        metric_logger.meters['loss'].global_avg,
        device=device, dtype=torch.float32
    )
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)

    return {
        'loss': avg_loss_tensor.item(),
        'learning_rate': optimizer.param_groups[0]['lr'],
    }


@torch.no_grad()
def validate_one_epoch(model, diffusion_trainer, loader, local_rank, use_bf16=True):
    model.eval()
    device = torch.device(f'cuda:{local_rank}')
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        data, conditions = batch
        data       = data.to(device, non_blocking=True)
        conditions = conditions.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_bf16):
            loss, _ = diffusion_trainer.get_loss(
                model, data, conditions, loss_in_fp32=use_bf16
            )

        total_loss += loss.item()
        n_batches  += 1

    avg_loss_tensor = torch.tensor(
        total_loss / max(n_batches, 1),
        device=device, dtype=torch.float32
    )
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
    return {'val_loss': avg_loss_tensor.item()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # Reproducibility — each rank gets its own seed offset for data sampling
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    use_bf16 = not args.no_mixed_precision
    batch_per_gpu = args.batch_size // world_size
    assert batch_per_gpu > 0, \
        f'batch_size ({args.batch_size}) must be >= world_size ({world_size})'

    # --- create save dir (rank 0 only) ---
    if is_main_process():
        os.makedirs(args.save_path, exist_ok=True)
        with open(os.path.join(args.save_path, 'args_fsdp.json'), 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)

    # -------------------------------------------------------------------
    # Dataset & DataLoader
    # -------------------------------------------------------------------
    collate_fn = image_collate_fn if args.train_mode == 'image' else pixel_collate_fn
    ds_kwargs = dict(
        data_path=args.data_path,
        train_mode=args.train_mode,
        eval_ratio=args.eval_ratio,
        data_format=args.train_mode,
        R_n=args.R_n,
    )
    if hasattr(args, 'sensor_down_sample_rate'):
        ds_kwargs['sensor_down_sample_rate'] = args.sensor_down_sample_rate

    DatasetCls = HFD_data if args.dataset == 'HFD' else HASCID_data
    train_dataset = DatasetCls(split='train', **ds_kwargs)
    eval_dataset  = DatasetCls(split='test',  **ds_kwargs)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=True, seed=args.seed
    )
    eval_sampler = DistributedSampler(
        eval_dataset, num_replicas=world_size, rank=rank,
        shuffle=False, seed=args.seed
    )

    nw = min(args.num_workers, batch_per_gpu, 4)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_per_gpu, sampler=train_sampler,
        num_workers=nw, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_per_gpu, sampler=eval_sampler,
        num_workers=nw, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
    )

    print_rank0(f'Train samples: {len(train_dataset)} | '
                f'Eval samples: {len(eval_dataset)} | '
                f'Batch/GPU: {batch_per_gpu} | '
                f'World size: {world_size}')

    # -------------------------------------------------------------------
    # Model — load checkpoint on CPU *before* FSDP wrapping
    # -------------------------------------------------------------------
    model = U2NetHyperspectral(
        spectral_channels=args.spectral_channels,
        sensor_channels=args.sensor_channels,
        base_channels=args.base_channels,
        use_channel_3d_conv=args.use_channel_3d_conv,
        channel_kernel=args.channel_kernel,
        spatial_kernel=args.spatial_kernel,
        channel_num_filters=args.channel_num_filters,
    )

    ckpt = load_checkpoint_to_cpu(args.resume)
    start_epoch = 0
    ckpt_optim_state = None
    ckpt_sched_state = None

    if ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        start_epoch    = ckpt.get('epoch', 0) + 1
        ckpt_optim_state = ckpt.get('optimizer_state_dict')
        ckpt_sched_state = ckpt.get('scheduler_state_dict')
        print_rank0(f'Resuming from epoch {start_epoch}  '
                    f'(ckpt loss: {ckpt.get("loss", "?")})')
        del ckpt

    # -------------------------------------------------------------------
    # Wrap with FSDP (moves model to GPU; sync_module_states broadcasts
    # rank-0 weights to all other ranks automatically)
    # -------------------------------------------------------------------
    model = wrap_model_with_fsdp(model, args, local_rank)

    n_params = sum(p.numel() for p in model.parameters())
    print_rank0(
        f'\nFSDP model ready — {n_params:,} params total\n'
        f'  sharding: {args.sharding_strategy}  '
        f'  bf16: {use_bf16}  '
        f'  cpu_offload: {args.cpu_offload}\n'
        f'  wrap_min_params: {args.wrap_min_params:,}  '
        f'use_channel_3d_conv: {args.use_channel_3d_conv}'
    )

    # -------------------------------------------------------------------
    # Diffusion trainer (lives on CPU, schedules are float32 tensors on GPU)
    # -------------------------------------------------------------------
    diffusion_trainer = DiffusionTrainer(
        n_timesteps=args.timesteps,
        loss_type=args.loss_type,
        prediction_type=args.prediction_type,
        beta_schedule=args.noise_schedule,
        device=device,
    )

    # -------------------------------------------------------------------
    # Optimizer & scheduler — created AFTER FSDP wrapping
    # -------------------------------------------------------------------
    adam_eps = 1e-6 if use_bf16 else 1e-8
    optimizer  = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=adam_eps,
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * args.lrf
    )

    # Restore optimizer / scheduler states if resuming
    if ckpt_optim_state is not None:
        # Convert from a full (consolidated) optim state dict to the
        # sharded representation expected by the FSDP-wrapped optimizer.
        sharded_optim = FSDP.optim_state_dict_to_load(
            model, optimizer, ckpt_optim_state
        )
        optimizer.load_state_dict(sharded_optim)
        print_rank0('Optimizer state restored.')
        del ckpt_optim_state

    for g in optimizer.param_groups:
        g['eps'] = adam_eps

    if ckpt_sched_state is not None:
        scheduler.load_state_dict(ckpt_sched_state)
        print_rank0(f'Scheduler state restored (last_epoch={scheduler.last_epoch}).')

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    train_history = []
    best_val_loss = float('inf')

    print_rank0('=' * 80)
    print_rank0(f'Starting training: epochs {start_epoch+1} → {args.num_epochs}')
    print_rank0('=' * 80)

    # Track the current sensor down-sample rate so we only log transitions.
    current_ds_rate = None

    for epoch in range(start_epoch, args.num_epochs):
        # Sensor-condition warmup: keep rate=1 for the first N epochs so the
        # model sees full-resolution conditioning while learning basic spectral
        # structure, then anneal to the target rate. Mutating the dataset here
        # is safe because persistent_workers defaults to False — DataLoader
        # workers are re-forked on the next iteration and inherit the new value.
        if epoch < args.ds_warmup_epochs:
            desired_ds = 1
        else:
            desired_ds = args.sensor_down_sample_rate
        if desired_ds != current_ds_rate:
            train_dataset.sensor_down_sample_rate = desired_ds
            eval_dataset.sensor_down_sample_rate = desired_ds
            print_rank0(f'[epoch {epoch+1}] sensor_down_sample_rate → {desired_ds}')
            current_ds_rate = desired_ds

        # DistributedSampler must know the epoch for reproducible shuffling
        train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_metrics = train_one_epoch(
            model, diffusion_trainer, train_loader, optimizer,
            epoch, local_rank, world_size,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            use_bf16=use_bf16,
            verbose_log=is_main_process(),
        )
        epoch_time = time.time() - t0

        scheduler.step()

        train_history.append(train_metrics)
        print_rank0(
            f'Epoch {epoch+1}/{args.num_epochs}  '
            f'loss: {train_metrics["loss"]:.6f}  '
            f'lr: {train_metrics["learning_rate"]:.2e}  '
            f'time: {epoch_time:.1f}s'
        )

        # Validation
        if args.val_every > 0 and (epoch + 1) % args.val_every == 0:
            eval_sampler.set_epoch(epoch)
            val_metrics = validate_one_epoch(
                model, diffusion_trainer, eval_loader, local_rank, use_bf16
            )
            print_rank0(f'  → val_loss: {val_metrics["val_loss"]:.6f}')

            if is_main_process() and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                save_fsdp_checkpoint(
                    model, optimizer, scheduler, epoch,
                    best_val_loss,
                    os.path.join(args.save_path, 'best_model.pth')
                )

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_fsdp_checkpoint(
                model, optimizer, scheduler, epoch,
                train_metrics['loss'],
                os.path.join(args.save_path, f'checkpoint_epoch_{epoch+1}.pth')
            )

    # Final checkpoint
    if is_main_process():
        save_fsdp_checkpoint(
            model, optimizer, scheduler, args.num_epochs - 1,
            train_history[-1]['loss'] if train_history else 0.0,
            os.path.join(args.save_path, 'final_model.pth')
        )

        # Save training history
        if train_history:
            header = ','.join(train_history[0].keys())
            rows   = [[m[k] for k in train_history[0]] for m in train_history]
            with open(os.path.join(args.save_path, 'train_history.txt'), 'w') as f:
                f.write(header + '\n')
                np.savetxt(f, np.array(rows), fmt='%.6f', delimiter=',')
            with open(os.path.join(args.save_path, 'train_history.json'), 'w') as f:
                json.dump(train_history, f, indent=2)

        losses = [h['loss'] for h in train_history]
        print('\nTraining complete!')
        print(f'  Epochs: {len(train_history)}')
        print(f'  Initial loss: {losses[0]:.6f}')
        print(f'  Final loss:   {losses[-1]:.6f}')
        print(f'  Best loss:    {min(losses):.6f}')

    cleanup_distributed()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _normalise_save_path(args):
    """Mirror the path-normalisation logic from main_2d.py."""
    parts = os.path.normpath(args.save_path).split(os.sep)

    if 'HFD' in args.data_path and 'HFD' not in parts[-3]:
        parts[-3] = 'HFD'
        args.save_path = os.path.join(*parts)
    elif 'HASCID' in args.data_path and 'HASCID' not in parts[-3]:
        parts[-3] = 'HASCID'
        args.save_path = os.path.join(*parts)

    parts = os.path.normpath(args.save_path).split(os.sep)
    r_folder = 'PH5' if args.R_n is None else f'R_{args.R_n}'
    if r_folder not in parts[-2]:
        parts[-2] = r_folder
        args.save_path = os.path.join(*parts)

    parts = os.path.normpath(args.save_path).split(os.sep)
    loss_folder = f'{args.loss_type}_loss'
    if loss_folder not in parts[-1]:
        parts[-1] = loss_folder
        args.save_path = os.path.join(*parts)

    if args.sensor_down_sample_rate > 1:
        parts = os.path.normpath(args.save_path).split(os.sep)
        ds_folder = f'down_sample_{args.sensor_down_sample_rate}'
        if ds_folder not in parts[-4]:
            parts[-4] = ds_folder
            args.save_path = os.path.join(*parts)


if __name__ == '__main__':
    parser = get_parser()
    args   = parser.parse_args()
    _normalise_save_path(args)

    # Only rank 0 prints the header (torchrun sets LOCAL_RANK before __main__)
    if int(os.environ.get('RANK', 0)) == 0:
        print('=' * 80)
        print('2D U-Net Diffusion Model — FSDP Multi-GPU Training')
        print('=' * 80)
        for k, v in vars(args).items():
            print(f'  {k}: {v}')
        print('=' * 80)

    main(args)


'''
3D conv kernel
Diffusion -> HxW R_response (H/4, W/4 -> H, W) -> 1D Diffusion reconstruct
HSI Drive set
'''