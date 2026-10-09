"""Time one training step of the residual model for the pilot and full-size configurations.

Single GPU:   python scripts/time_train_step.py --base_channels 64 --batch_size 8 --device cuda:0
FSDP, 2 GPUs: torchrun --standalone --nproc_per_node=2 scripts/time_train_step.py --base_channels 256 --batch_size 8 --fsdp
Checks nvidia-smi free memory first and refuses to run on a GPU with < 3 GB free.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.hfd_residual import HFDResidualData, residual_collate_fn  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from model.residual_wrapper import ResidualWrapper  # noqa: E402
from model.u2net_hyperspectral import U2NetHyperspectral  # noqa: E402

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')
PRIOR = os.path.join(REPO, 'results', 'residual_warmstart', 'linear_prior_R1.npz')


def free_mb(index):
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', f'--id={index}'],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_channels', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--d', type=int, default=4)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--fsdp', action='store_true')
    args = ap.parse_args()

    if args.fsdp:
        torch.distributed.init_process_group('nccl')
        rank = torch.distributed.get_rank(); local = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local}')
    else:
        rank, local = 0, int(args.device.split(':')[-1]); device = torch.device(args.device)
    if free_mb(local) < 3000:
        raise SystemExit(f'GPU {local} has only {free_mb(local)} MiB free; refusing to run')
    torch.cuda.set_device(device)

    ds = HFDResidualData(data_path=DATA_ROOT, split='train', sensor_down_sample_rate=args.d, prior_path=PRIOR)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=residual_collate_fn)
    net = U2NetHyperspectral(spectral_channels=64, sensor_channels=30, base_channels=args.base_channels, extra_in_channels=64)
    model = ResidualWrapper(net, use_x0_hat=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if args.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools
        model = FSDP(model, auto_wrap_policy=functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000),
                     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
                     device_id=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = DiffusionTrainer(device=device, prediction_type='v', loss_type='l1', snr_gamma=5.0)

    times = []
    it = iter(loader)
    for step in range(args.steps + 3):
        x, y_d, xh = next(it)
        x, y_d, xh = x.to(device), y_d.to(device), xh.to(device)
        r = (x - xh) / ds.sigma_d
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = trainer.get_loss(model, r, {'sensor': y_d, 'x0_hat': xh}, loss_in_fp32=True)
        opt.zero_grad(set_to_none=True); loss.backward()
        (model.clip_grad_norm_(1.0) if args.fsdp else torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step(); torch.cuda.synchronize()
        if step >= 3: times.append(time.time() - t0)
    sec = sum(times) / len(times); peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
    if rank == 0:
        n_train = len(ds); world = torch.distributed.get_world_size() if args.fsdp else 1
        epoch_sec = sec * n_train / (args.batch_size * world)
        rec = dict(base_channels=args.base_channels, batch_size=args.batch_size, fsdp=args.fsdp, world=world, params=n_params,
                   sec_per_step=sec, peak_gb=peak, sec_per_epoch=epoch_sec, loss=float(loss))
        print(json.dumps(rec, indent=1))
        path = os.path.join(REPO, 'results', 'residual_warmstart', 'timing.json')
        allrec = json.load(open(path)) if os.path.exists(path) else []
        allrec.append(rec); json.dump(allrec, open(path, 'w'), indent=1)
    if args.fsdp:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
