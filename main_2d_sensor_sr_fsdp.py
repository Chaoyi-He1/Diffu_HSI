"""Sensor super-resolution diffusion — FSDP multi-GPU training.

Runs a separate diffusion model (U2NetSensorSR) whose target is the
full-resolution sensor measurement and whose conditioning is a
spatially-downsampled version of the same measurement.

Launch (single node, 2 GPUs):
    torchrun --standalone --nproc_per_node=2 main_2d_sensor_sr_fsdp.py [args]
"""
import argparse
import functools
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim

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

from model.diffusion_trainer import DiffusionTrainer
from model.u2net_sensor_sr import U2NetSensorSR
from data_loader.HFD_sensor_sr_dataset import (
    HFD_SensorSR_data, sensor_sr_image_collate_fn,
)
from train_eval.train_sensor_sr import train_one_epoch, validate_one_epoch


def get_parser():
    p = argparse.ArgumentParser(
        description='Sensor Super-Resolution Diffusion — FSDP Training'
    )
    # dataset
    p.add_argument('--data_path', type=str,
                   default='dataset/HFD100 Mat dataset')
    p.add_argument('--type', type=str, default='Flower',
                   choices=['Flower', 'Leaves', 'Scenses'])
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--eval_ratio', type=float, default=0.1)
    p.add_argument('--R-n', type=int, default=1, dest='R_n')
    p.add_argument('--ds_sr', type=int, default=8,
                   help='Spatial downsample rate for sensor conditioning.')
    p.add_argument('--sr_downsample_method', type=str, default='strided',
                   choices=['strided', 'avg_pool'])
    p.add_argument('--sensor_stats_path', type=str, required=True,
                   help='JSON file produced by scripts/compute_sensor_stats.py')

    # model
    p.add_argument('--sensor_channels', type=int, default=30)
    p.add_argument('--base_channels', type=int, default=256)
    p.add_argument('--use_channel_3d_conv', action='store_true')
    p.add_argument('--channel_kernel', type=int, default=7)
    p.add_argument('--spatial_kernel', type=int, default=3)
    p.add_argument('--channel_num_filters', type=int, default=4)

    # optimisation
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--lrf', type=float, default=0.033)
    p.add_argument('--grad_clip', type=float, default=1.0)

    # training schedule
    p.add_argument('--val_every', type=int, default=100)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--log_interval', type=int, default=200)

    # diffusion
    p.add_argument('--loss_type', type=str, default='l1',
                   choices=['l1', 'l2'])
    p.add_argument('--noise_schedule', type=str, default='linear',
                   choices=['linear'])
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--prediction_type', type=str, default='v',
                   choices=['eps', 'x0', 'v'])

    # FSDP
    p.add_argument('--sharding_strategy', type=str, default='FULL_SHARD',
                   choices=['FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD'])
    p.add_argument('--no_mixed_precision', action='store_true')
    p.add_argument('--cpu_offload', action='store_true')
    p.add_argument('--wrap_min_params', type=int, default=1_000_000)

    # output / resume
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--save_path', type=str, required=True)
    p.add_argument('--seed', type=int, default=42)
    return p


def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def is_main_process():
    return dist.get_rank() == 0


def print_rank0(*a, **k):
    if is_main_process():
        print(*a, **k)


def save_fsdp_checkpoint(model, optimizer, scheduler, epoch, loss, save_path):
    rank = dist.get_rank()
    model_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, model_cfg):
        model_state = model.state_dict()
    optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                               optim_state_dict_config=optim_cfg):
        optim_state = FSDP.optim_state_dict(model, optimizer)
    if rank == 0:
        ckpt = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model_state,
            'optimizer_state_dict': optim_state,
            'scheduler_state_dict':
                scheduler.state_dict() if scheduler is not None else None,
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(ckpt, save_path)
        print(f'[rank 0] Checkpoint saved → {save_path}')
    dist.barrier()


def load_checkpoint_to_cpu(resume_path):
    if resume_path and resume_path.endswith('.pth') \
            and os.path.exists(resume_path):
        print_rank0(f'Loading checkpoint from {resume_path} …')
        return torch.load(resume_path, map_location='cpu',
                          weights_only=False)
    return None


def wrap_model_with_fsdp(model, args, local_rank):
    strategy_map = {
        'FULL_SHARD': ShardingStrategy.FULL_SHARD,
        'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
        'NO_SHARD': ShardingStrategy.NO_SHARD,
    }
    if not args.no_mixed_precision:
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp = None
    policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=args.wrap_min_params
    )
    cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None
    return FSDP(
        model,
        sharding_strategy=strategy_map[args.sharding_strategy],
        mixed_precision=mp,
        auto_wrap_policy=policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=cpu_offload,
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=True,
    )


def main(args):
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    use_bf16 = not args.no_mixed_precision
    batch_per_gpu = args.batch_size // world_size
    assert batch_per_gpu > 0, \
        f'batch_size ({args.batch_size}) must be >= world_size ({world_size})'

    if is_main_process():
        os.makedirs(args.save_path, exist_ok=True)
        with open(os.path.join(args.save_path, 'args_sensor_sr.json'),
                  'w') as f:
            json.dump(vars(args), f, indent=2, default=str)

    # Dataset / loaders
    ds_kwargs = dict(
        data_path=args.data_path,
        stats_path=args.sensor_stats_path,
        eval_ratio=args.eval_ratio,
        type=args.type,
        R_n=args.R_n,
        sr_downsample_rate=args.ds_sr,
        sr_downsample_method=args.sr_downsample_method,
    )
    train_ds = HFD_SensorSR_data(split='train', **ds_kwargs)
    eval_ds = HFD_SensorSR_data(split='test', **ds_kwargs)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank,
        shuffle=True, seed=args.seed,
    )
    eval_sampler = DistributedSampler(
        eval_ds, num_replicas=world_size, rank=rank,
        shuffle=False, seed=args.seed,
    )

    nw = min(args.num_workers, batch_per_gpu, 4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_per_gpu, sampler=train_sampler,
        num_workers=nw, collate_fn=sensor_sr_image_collate_fn,
        pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_per_gpu, sampler=eval_sampler,
        num_workers=nw, collate_fn=sensor_sr_image_collate_fn,
        pin_memory=True, drop_last=False,
    )

    print_rank0(
        f'Train: {len(train_ds)}  Eval: {len(eval_ds)}  '
        f'Batch/GPU: {batch_per_gpu}  World: {world_size}  '
        f'ds_sr: {args.ds_sr} ({args.sr_downsample_method})'
    )

    # Model
    model = U2NetSensorSR(
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
        start_epoch = ckpt.get('epoch', 0) + 1
        ckpt_optim_state = ckpt.get('optimizer_state_dict')
        ckpt_sched_state = ckpt.get('scheduler_state_dict')
        print_rank0(f'Resuming from epoch {start_epoch} '
                    f'(ckpt loss: {ckpt.get("loss", "?")})')
        del ckpt

    model = wrap_model_with_fsdp(model, args, local_rank)
    n_params = sum(p.numel() for p in model.parameters())
    print_rank0(
        f'\nFSDP model ready — {n_params:,} params total\n'
        f'  sharding: {args.sharding_strategy}  bf16: {use_bf16}  '
        f'cpu_offload: {args.cpu_offload}\n'
        f'  use_channel_3d_conv: {args.use_channel_3d_conv}  '
        f'base_channels: {args.base_channels}'
    )

    diffusion_trainer = DiffusionTrainer(
        n_timesteps=args.timesteps,
        loss_type=args.loss_type,
        prediction_type=args.prediction_type,
        beta_schedule=args.noise_schedule,
        device=device,
    )

    adam_eps = 1e-6 if use_bf16 else 1e-8
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, eps=adam_eps,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * args.lrf
    )

    if ckpt_optim_state is not None:
        sharded = FSDP.optim_state_dict_to_load(
            model, optimizer, ckpt_optim_state
        )
        optimizer.load_state_dict(sharded)
        print_rank0('Optimizer state restored.')
        del ckpt_optim_state

    for g in optimizer.param_groups:
        g['eps'] = adam_eps

    if ckpt_sched_state is not None:
        scheduler.load_state_dict(ckpt_sched_state)
        print_rank0(f'Scheduler restored (last_epoch={scheduler.last_epoch}).')

    print_rank0('=' * 80)
    print_rank0(f'Starting training: epochs {start_epoch + 1} → {args.num_epochs}')
    print_rank0('=' * 80)

    train_history = []
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        t0 = time.time()
        tm = train_one_epoch(
            model, diffusion_trainer, train_loader, optimizer,
            epoch, local_rank, world_size,
            grad_clip=args.grad_clip, log_interval=args.log_interval,
            use_bf16=use_bf16, verbose_log=is_main_process(),
        )
        dt = time.time() - t0
        scheduler.step()
        train_history.append(tm)
        print_rank0(
            f'Epoch {epoch+1}/{args.num_epochs}  '
            f'loss: {tm["loss"]:.6f}  lr: {tm["learning_rate"]:.2e}  '
            f'time: {dt:.1f}s'
        )

        if args.val_every > 0 and (epoch + 1) % args.val_every == 0:
            eval_sampler.set_epoch(epoch)
            vm = validate_one_epoch(
                model, diffusion_trainer, eval_loader, local_rank, use_bf16
            )
            print_rank0(f'  → val_loss: {vm["val_loss"]:.6f}')
            # All ranks see the same averaged val_loss (all_reduce in
            # validate_one_epoch), so the decision to save is consistent.
            # save_fsdp_checkpoint itself runs FSDP collectives — must be
            # called by every rank, not gated on is_main_process().
            if vm['val_loss'] < best_val_loss:
                best_val_loss = vm['val_loss']
                save_fsdp_checkpoint(
                    model, optimizer, scheduler, epoch, best_val_loss,
                    os.path.join(args.save_path, 'best_model.pth'),
                )

        if (epoch + 1) % args.save_every == 0:
            save_fsdp_checkpoint(
                model, optimizer, scheduler, epoch, tm['loss'],
                os.path.join(args.save_path,
                             f'checkpoint_epoch_{epoch+1}.pth'),
            )

    # save_fsdp_checkpoint runs FSDP collectives; call on every rank.
    save_fsdp_checkpoint(
        model, optimizer, scheduler, args.num_epochs - 1,
        train_history[-1]['loss'] if train_history else 0.0,
        os.path.join(args.save_path, 'final_model.pth'),
    )
    if is_main_process():
        if train_history:
            header = ','.join(train_history[0].keys())
            rows = [[m[k] for k in train_history[0]] for m in train_history]
            with open(os.path.join(args.save_path, 'train_history.txt'),
                      'w') as f:
                f.write(header + '\n')
                np.savetxt(f, np.array(rows), fmt='%.6f', delimiter=',')
            with open(os.path.join(args.save_path, 'train_history.json'),
                      'w') as f:
                json.dump(train_history, f, indent=2)
        losses = [h['loss'] for h in train_history]
        print('\nTraining complete!')
        if losses:
            print(f'  Initial loss: {losses[0]:.6f}')
            print(f'  Final   loss: {losses[-1]:.6f}')
            print(f'  Best    loss: {min(losses):.6f}')

    dist.destroy_process_group()


if __name__ == '__main__':
    args = get_parser().parse_args()
    if int(os.environ.get('RANK', 0)) == 0:
        print('=' * 80)
        print('Sensor Super-Resolution Diffusion — FSDP Multi-GPU Training')
        print('=' * 80)
        for k, v in vars(args).items():
            print(f'  {k}: {v}')
        print('=' * 80)
    main(args)
