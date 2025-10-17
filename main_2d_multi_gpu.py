import torch
import misc.util as utils
from model.u2net_hyperspectral import *
from model.diffusion_trainer import *
import argparse
import os
import time
import numpy as np
import random
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing
from train_eval.train_2d import *
from data_loader.my_dataset import HASCID_data, pixel_collate_fn, image_collate_fn
from data_loader.HFD_dataset import HFD_data

torch.multiprocessing.set_sharing_strategy('file_system')


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='2D U-Net Diffusion Model for Direct Hyperspectral Image Reconstruction')
    
    # dataset parameters
    parser.add_argument('--data_path', type=str, default='dataset/HFD100 Mat dataset', choices=['dataset/HASCID-Dataset',
                                                                                            'dataset/HFD100 Mat dataset'], help='path to dataset')
    parser.add_argument('--train_mode', type=str, default='image', choices=['pixel', 'image'], 
                        help='training mode, if pixel, then randomly sample pixels from all training images; \
                             if image, then randomly sample images')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to load data')
    parser.add_argument('--batch_size', type=int, default=6, help='input batch size for training (smaller for direct training)')
    parser.add_argument('--eval_ratio', type=float, default=0.1, help='the ratio of test data during training')
    parser.add_argument('--R-n', type=int, default=1, help='the number of random measurements, if None, then use full measurements')
    parser.add_argument('--dataset', type=str, choices=['HASCID', 'HFD'], default='HFD', help='which dataset to use')
    
    # model parameters
    parser.add_argument('--spectral_channels', type=int, default=64, help='number of spectral bands of input hyperspectral data')
    parser.add_argument('--sensor_channels', type=int, default=30, help='number of channels of the sensor response (condition)')
    parser.add_argument('--base_channels', type=int, default=128, help='base channels of diffusion U-Net model (smaller for direct training)')
    
    # optimization parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (smaller for direct training)')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--lrf', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping threshold')
    parser.add_argument('--scaler', type=str, default='amp', choices=['none', 'amp'], help='use automatic mixed precision training')
    
    # training schedule parameters
    parser.add_argument('--val_every', type=int, default=1000, help='validate every N epochs')
    parser.add_argument('--save_every', type=int, default=50, help='save checkpoint every N epochs')
    parser.add_argument('--generate_every', type=int, default=2000, help='generate samples every N epochs')
    parser.add_argument('--log_interval', type=int, default=200, help='log metrics every N steps')
    
    # diffusion trainer parameters
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'l2', 'huber'], 
                        help='loss type for diffusion training')
    parser.add_argument('--noise_schedule', type=str, default='linear', choices=['linear', 'cosine'],
                        help='noise schedule for diffusion process')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of diffusion timesteps')
    parser.add_argument('--prediction_type', type=str, default='eps', choices=['eps', 'x0', 'v'],
                        help='diffusion prediction type')
    
    # output parameters
    parser.add_argument('--resume', type=str, default='results/2d_hsi_diffusion/HFD/R_1/l1_loss/checkpoint_epoch_51.pth', help='path to resume diffusion model checkpoint')
    parser.add_argument('--save_path', type=str, default='results/2d_hsi_diffusion/HFD/R_1/l1_loss', help='path to save results')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for computation')
    
    # distributed training parameter
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    
    return parser


def main(args):
    utils.init_distributed_mode(args)
    
    # Only save arguments on main process and create directory
    if utils.is_main_process():
        os.makedirs(args.save_path, exist_ok=True)
        import json
        with open(os.path.join(args.save_path, 'args_2d_multi_gpu.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    assert args.batch_size % args.world_size == 0, "--batch-size must be divisible by number of GPUs"
        
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device based on distributed setup
    if hasattr(args, 'gpu') and args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    args.device = device
    print(f"Using device: {device} (rank {args.rank})")
    
    # create dataset and dataloader
    if args.dataset == 'HASCID':
        train_dataset = HASCID_data(data_path=args.data_path, 
                                    train_mode=args.train_mode, 
                                    split='train', 
                                    eval_ratio=args.eval_ratio, 
                                    data_format=args.train_mode,
                                    R_n=args.R_n)
        eval_dataset = HASCID_data(data_path=args.data_path,
                                    train_mode=args.train_mode, 
                                    split='test', 
                                    eval_ratio=args.eval_ratio, 
                                    data_format=args.train_mode,
                                    R_n=args.R_n)
    elif args.dataset == 'HFD':
        train_dataset = HFD_data(data_path=args.data_path,
                                 train_mode=args.train_mode, 
                                 split='train', 
                                 eval_ratio=args.eval_ratio, 
                                 data_format=args.train_mode,
                                 R_n=args.R_n)
        eval_dataset = HFD_data(data_path=args.data_path,
                                train_mode=args.train_mode, 
                                split='test', 
                                eval_ratio=args.eval_ratio, 
                                data_format=args.train_mode,
                                R_n=args.R_n)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Create distributed samplers
    sampler_train = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=args.rank,
        shuffle=True,
        seed=args.seed
        )
    sampler_eval = torch.utils.data.distributed.DistributedSampler(
        eval_dataset,
        num_replicas=dist.get_world_size(),
        rank=args.rank,
        shuffle=False,
        seed=args.seed
        )
    
    nw = min([os.cpu_count(), int(args.batch_size // dist.get_world_size()) if int(args.batch_size // dist.get_world_size()) > 1 else 0, 2])  # number of workers
    if args.rank in [-1, 0]:
        print(f"Using {nw} dataloader workers per GPU (total {nw * dist.get_world_size()})")
    
    if args.train_mode == 'pixel':
        train_loader = DataLoader(train_dataset, 
                                  batch_size=int(args.batch_size // dist.get_world_size()), 
                                  sampler=sampler_train,
                                  num_workers=nw, 
                                  collate_fn=pixel_collate_fn)
        eval_loader = DataLoader(eval_dataset, 
                                 batch_size=int(args.batch_size // dist.get_world_size()), 
                                 sampler=sampler_eval,
                                 num_workers=nw, 
                                 collate_fn=pixel_collate_fn)
    else:
        # Use image collate function for proper 2D image handling
        train_loader = DataLoader(train_dataset, 
                                  batch_size=int(args.batch_size // dist.get_world_size()), 
                                  sampler=sampler_train,
                                  num_workers=nw, 
                                  collate_fn=image_collate_fn)
        eval_loader = DataLoader(eval_dataset, 
                                 batch_size=int(args.batch_size // dist.get_world_size()), 
                                 sampler=sampler_eval,
                                 num_workers=nw, 
                                 collate_fn=image_collate_fn)

    if args.rank in [-1, 0]:
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of evaluation samples: {len(eval_dataset)}")
    
    # Create diffusion model - operates directly on hyperspectral data
    model = U2NetHyperspectral(
        spectral_channels=args.spectral_channels,
        sensor_channels=args.sensor_channels,
        base_channels=args.base_channels,
    )
    
    # Create diffusion trainer
    trainer = DiffusionTrainer(
        n_timesteps=args.timesteps,
        loss_type=args.loss_type,
        prediction_type=args.prediction_type,
        beta_schedule=args.noise_schedule,
        device=device
    )
    
    # load trained diffusion model if exists
    start_epoch = 0
    if args.resume and args.resume.endswith('.pth') and os.path.exists(args.resume):
        print(f"Loading diffusion model from {args.resume} (rank {args.rank})")
            
        ckpt = torch.load(args.resume, weights_only=False, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        
        # Verify model loading
        for k, v in model.named_parameters():
            if not torch.equal(v, ckpt['model_state_dict'][k]):
                print(f"Parameter {k} not loaded correctly.")
                raise ValueError(f"Parameter {k} not loaded correctly.")
                print(f"Parameter {k} not loaded correctly.")
                raise ValueError(f"Parameter {k} not loaded correctly.")
        if args.rank in [-1, 0]:
            print(f"Diffusion model loaded successfully from {args.resume}")
        
        # if 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict'] is not None:
        #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        #     print("Optimizer state loaded.")
        
        # if 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
        #     lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        #     print("LR scheduler state loaded.")
        # elif 'lr_scheduler' in ckpt and ckpt['lr_scheduler'] is not None:
        #     lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        #     print("LR scheduler state loaded.")
        
        # start_epoch = ckpt['epoch'] + 1 if 'epoch' in ckpt else 0
        # lr_scheduler.last_epoch = start_epoch - 1  # Adjust for lr_scheduler step
        # print(f"Resuming training from epoch {start_epoch}")
        
        del ckpt  # free memory
        torch.cuda.empty_cache() # clear cache
    else:
        if args.resume and args.rank in [-1, 0]:
            print(f"Checkpoint file {args.resume} not found, training from scratch.")
        elif args.rank in [-1, 0]:
            print("No diffusion model checkpoint specified, training from scratch.")
    
    # Move model to device before wrapping with DDP
    model = model.to(device)
    
    # Wrap model with DistributedDataParallel
    # Enhanced configuration to handle gradient stride mismatches in U-Net architectures
    if args.distributed:
        # For U-Net models with complex layer patterns, we use the most conservative settings
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu], 
            output_device=args.gpu,
            find_unused_parameters=False,
            gradient_as_bucket_view=False,  # Disable bucket views to avoid stride issues
            bucket_cap_mb=10,               # Very small bucket size for U-Net compatibility
            # Note: static_graph=True removed as it can cause issues with dynamic U-Net structures
        )
    else:
        # For single GPU fallback
        pass
    
    optimizer = torch.optim.AdamW(model.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * args.lrf)
    scaler = torch.amp.GradScaler(enabled=(args.scaler == 'amp'))
    
    lr_scheduler.last_epoch = start_epoch - 1  # Adjust for lr_scheduler step
    
    if args.rank in [-1, 0]:
        print(f"\nModel Architecture:")
        print(f"Direct 2D Hyperspectral Diffusion U-Net (Multi-GPU: {args.distributed})")
        print(f"Spectral channels: {args.spectral_channels}")
        print(f"Sensor condition channels: {args.sensor_channels}")
        print(f"Base channels: {args.base_channels}")
        print(f"Total model parameters: {sum(p.numel() for p in model.module.parameters() if p.requires_grad):,}")
        print(f"World size: {args.world_size}, Effective batch size: {args.batch_size}")
    
    # Calculate memory usage estimate with per-GPU batch size
    batch_size_per_gpu = int(args.batch_size // args.world_size)
    sample_input = torch.randn(batch_size_per_gpu, args.spectral_channels, 64, 64).to(device)
    sample_condition = torch.randn(batch_size_per_gpu, args.sensor_channels, 64, 64).to(device)
    sample_time = torch.randint(0, args.timesteps, (batch_size_per_gpu,)).to(device)
    
    model.eval()
    with torch.no_grad():
        try:
            _ = model(sample_input, sample_condition, sample_time)
            print(f"Model forward pass successful with batch size {batch_size_per_gpu} on GPU {args.rank}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: Batch size {batch_size_per_gpu} may be too large for GPU {args.rank}")
                print("Consider reducing batch_size or base_channels")
            raise e
    model.train()
    
    # start training with the training functions from train_2d.py
    # Only save on main process
    save_dir = args.save_path if utils.is_main_process() else None
    
    trained_model, train_history = train_full_pipeline(
        model=model,
        diffusion_trainer=trainer,
        train_dataloader=train_loader,
        val_dataloader=eval_loader,
        num_epochs=args.num_epochs,
        device=args.device,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        save_dir=save_dir,
        grad_clip=args.grad_clip,
        val_every=args.val_every,
        save_every=args.save_every,
        generate_every=args.generate_every,
        scaler=scaler,
        log_interval=args.log_interval
    )
    
    # save the training history as .txt file (only on main process)
    if train_history and utils.is_main_process():
        # Create header with metric names
        header = ','.join(train_history[0].keys())
        
        # Convert dictionaries to arrays of values
        data_rows = []
        for epoch_metrics in train_history:
            row = [epoch_metrics[key] for key in train_history[0].keys()]
            data_rows.append(row)
        
        # Save with header
        with open(os.path.join(args.save_path, 'train_history.txt'), 'w') as f:
            f.write(header + '\n')
            np.savetxt(f, np.array(data_rows), fmt='%.6f', delimiter=',')
        
        # Also save as JSON for easier parsing
        import json
        with open(os.path.join(args.save_path, 'train_history.json'), 'w') as f:
            json.dump(train_history, f, indent=2)
    
    if utils.is_main_process():
        print(f"\nTraining completed!")
        print(f"Results saved to: {args.save_path}")
        print(f"Final training loss: {train_history[-1]['loss']:.6f}" if train_history else "No training history available")
        print(f"Checkpoints and generated samples saved in: {args.save_path}")
        
        # Print summary statistics
        if train_history:
            losses = [h['loss'] for h in train_history]
            print(f"\nTraining Summary:")
            print(f"  Epochs trained: {len(train_history)}")
            print(f"  Initial loss: {losses[0]:.6f}")
            print(f"  Final loss: {losses[-1]:.6f}")
            print(f"  Best loss: {min(losses):.6f}")
            print(f"  Loss improvement: {losses[0] - losses[-1]:.6f}")
            
            # Print memory and performance info
            if 'samples_per_sec' in train_history[-1]:
                avg_samples_per_sec = np.mean([h['samples_per_sec'] for h in train_history if 'samples_per_sec' in h])
                print(f"  Average training speed: {avg_samples_per_sec:.2f} samples/sec")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    # Sanity checks for args
    if "HASCID" in args.data_path:
        if args.dataset != 'HASCID':
            args.dataset = 'HASCID'
            # Make the print to be red text
            print(f"\033[91mWarning: dataset argument changed to 'HASCID' to match data_path.\033[0m", flush=True) 
        # check if HASCID is the third last folder in the save_path
        if "HASCID" not in os.path.normpath(args.save_path).split(os.sep)[-3]:
            # replace the second last folder with HASCID
            args.save_path = os.path.join(*os.path.normpath(args.save_path).split(os.sep)[:-3], 'HASCID', os.path.normpath(args.save_path).split(os.sep)[-2:])
            print(f"\033[91mWarning: save_path argument changed to include 'HASCID' folder.\033[0m", flush=True)
    elif "HFD" in args.data_path:
        if args.dataset != 'HFD':
            args.dataset = 'HFD'
            print(f"\033[91mWarning: dataset argument changed to 'HFD' to match data_path.\033[0m", flush=True)
        if "HFD" not in os.path.normpath(args.save_path).split(os.sep)[-3]:
            args.save_path = os.path.join(*os.path.normpath(args.save_path).split(os.sep)[:-3], 'HFD', os.path.normpath(args.save_path).split(os.sep)[-2:])
            print(f"\033[91mWarning: save_path argument changed to include 'HFD' folder.\033[0m", flush=True)
    
    if args.R_n is None:
        # check if "PH5" is in the second last folder of save_path
        if "PH5" not in os.path.normpath(args.save_path).split(os.sep)[-2]:
            # replace the last folder with PH5
            args.save_path = os.path.join(*os.path.normpath(args.save_path).split(os.sep)[:-2], 'PH5', os.path.normpath(args.save_path).split(os.sep)[-1])
            print(f"\033[91mWarning: save_path argument changed to include 'PH5' folder for full measurements.\033[0m", flush=True)
    elif args.R_n == 1:
        if "R_1" not in os.path.normpath(args.save_path).split(os.sep)[-2]:
            args.save_path = os.path.join(*os.path.normpath(args.save_path).split(os.sep)[:-2], 'R_1', os.path.normpath(args.save_path).split(os.sep)[-1])
            print(f"\033[91mWarning: save_path argument changed to include 'R_1' folder for R_n=1.\033[0m", flush=True)
    elif args.R_n == 2:
        if "R_2" not in os.path.normpath(args.save_path).split(os.sep)[-2]:
            args.save_path = os.path.join(*os.path.normpath(args.save_path).split(os.sep)[:-2], 'R_2', os.path.normpath(args.save_path).split(os.sep)[-1])
            print(f"\033[91mWarning: save_path argument changed to include 'R_2' folder for R_n=2.\033[0m", flush=True)
    else:
        # Raise error if R_n is not 1, 2, or None
        raise ValueError(f"Unsupported R_n value: {args.R_n}. Supported values are 1, 2, or None for full measurements.")
    
    if args.loss_type == 'l1':
        if 'l1_loss' not in os.path.normpath(args.save_path).split(os.sep)[-1]:
            args.save_path = os.path.join(*os.path.normpath(args.save_path).split(os.sep)[:-1], 'l1_loss')
            print(f"\033[91mWarning: save_path argument changed to include 'l1_loss' folder for L1 loss.\033[0m", flush=True)
    elif args.loss_type == 'l2':
        if 'l2_loss' not in os.path.normpath(args.save_path).split(os.sep)[-1]:
            args.save_path = os.path.join(*os.path.normpath(args.save_path).split(os.sep)[:-1], 'l2_loss')
            print(f"\033[91mWarning: save_path argument changed to include 'l2_loss' folder for L2 loss.\033[0m", flush=True)
    
    # print all arguments when on main process
    if utils.is_main_process():
        print("="*80)
        print("2D U-Net Diffusion Model for Direct Hyperspectral Reconstruction (Multi-GPU)")
        print("="*80)
        print("Arguments:")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        print("="*80)
        
        # Print warning about memory requirements
        print("WARNING: Multi-GPU hyperspectral diffusion requires significant GPU memory.")
        print("If you encounter out-of-memory errors, consider:")
        print("  - Reducing --batch_size per GPU (current per-GPU batch size: {})".format(int(args.batch_size // args.world_size)))
        print("  - Reducing --base_channels (current: {})".format(args.base_channels))
        print("  - Using mixed precision training (--scaler amp, default)")
        print("  - Using the VAE-based approach (main_2d_with_vae.py) for large datasets")
        print("="*80)
    
    main(args)