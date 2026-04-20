"""One-epoch train / validate helpers for sensor-SR diffusion.

The FSDP entrypoint imports these. They are thin copies of the helpers
inside main_2d_fsdp.py with no semantic changes; isolating them makes
the entrypoint easier to read and the helpers reusable.
"""
import torch
import torch.distributed as dist

from misc.util import MetricLogger, SmoothedValue


def train_one_epoch(
    model, diffusion_trainer, loader, optimizer, epoch,
    local_rank, world_size, grad_clip, log_interval,
    use_bf16=True, verbose_log=True,
):
    model.train()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch + 1}]'

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    device = torch.device(f'cuda:{local_rank}')

    for step, (data, cond) in enumerate(
        metric_logger.log_every(loader, log_interval, header, verbose=verbose_log)
    ):
        data = data.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_bf16):
            loss, _ = diffusion_trainer.get_loss(
                model, data, cond, loss_in_fp32=use_bf16
            )

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            model.clip_grad_norm_(grad_clip)

        optimizer.step()

        metric_logger.update(loss=loss.item(),
                              lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['loss'].update(loss.item(), n=data.shape[0])

    avg = torch.tensor(metric_logger.meters['loss'].global_avg,
                       device=device, dtype=torch.float32)
    dist.all_reduce(avg, op=dist.ReduceOp.AVG)
    return {
        'loss': avg.item(),
        'learning_rate': optimizer.param_groups[0]['lr'],
    }


@torch.no_grad()
def validate_one_epoch(model, diffusion_trainer, loader, local_rank,
                       use_bf16=True):
    model.eval()
    device = torch.device(f'cuda:{local_rank}')
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    total, n = 0.0, 0
    for data, cond in loader:
        data = data.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_bf16):
            loss, _ = diffusion_trainer.get_loss(
                model, data, cond, loss_in_fp32=use_bf16
            )
        total += loss.item()
        n += 1
    avg = torch.tensor(total / max(n, 1), device=device, dtype=torch.float32)
    dist.all_reduce(avg, op=dist.ReduceOp.AVG)
    return {'val_loss': avg.item()}
