import torch
import misc.util as utils
from model.latent_u2net_hyperspectral import *
from model.hyperspectral_vae import *
from model.diffusion_trainer import *
import argparse
import os
import time
import numpy as np
import random
from torch.utils.data import DataLoader
from train_eval.train_2d_with_vae import *
from data_loader.my_dataset import HASCID_data, pixel_collate_fn, image_collate_fn


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='2D U-Net Diffusion Model with VAE compression for Hyperspectral Image Reconstruction')
    
    # dataset parameters
    parser.add_argument('--data_path', type=str, default='dataset/HASCID-Dataset', help='path to dataset')
    parser.add_argument('--train_mode', type=str, default='image', choices=['pixel', 'image'], 
                        help='training mode, if pixel, then randomly sample pixels from all training images; \
                             if image, then randomly sample images')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to load data')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training')
    parser.add_argument('--eval_ratio', type=float, default=0.1, help='the ratio of test data during training')
    
    # model parameters
    # VAE parameters
    parser.add_argument('--vae_latent_channels', type=int, default=12, help='dimension of latent space of VAE compression')
    parser.add_argument('--vae_base_channels', type=int, default=128, help='base channels of VAE model')
    parser.add_argument('--vae_checkpoint', type=str, default='results/vae_hyperspectral/base_128_latent_12/vae_final_model.pth', 
                        help='path to pre-trained VAE model')
    
    # Diffusion model parameters
    parser.add_argument('--input_channels', type=int, default=160, help='number of spectral bands of input data')
    parser.add_argument('--sensor_channels', type=int, default=30, help='number of channels of the sensor response (condition)')
    parser.add_argument('--diffusion_base_channels', type=int, default=128, help='base channels of diffusion U-Net model')
    
    # optimization parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=2000, help='number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--lrf', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='gradient clipping threshold')
    parser.add_argument('--scaler', type=str, default='amp', choices=['none', 'amp'], help='use automatic mixed precision training')
    
    # training schedule parameters
    parser.add_argument('--val_every', type=int, default=1000, help='validate every N epochs')
    parser.add_argument('--save_every', type=int, default=100, help='save checkpoint every N epochs')
    parser.add_argument('--generate_every', type=int, default=1000, help='generate samples every N epochs')
    parser.add_argument('--log_interval', type=int, default=100, help='log metrics every N steps')
    
    # diffusion trainer parameters
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'l2', 'huber'], 
                        help='loss type for diffusion training')
    parser.add_argument('--noise_schedule', type=str, default='linear', choices=['linear', 'cosine'],
                        help='noise schedule for diffusion process')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of diffusion timesteps')
    
    # output parameters
    parser.add_argument('--resume', type=str, default='', help='path to resume diffusion model checkpoint')
    parser.add_argument('--save_path', type=str, default='results/2d_hsi_diffusion_vae', help='path to save results')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for computation')
    
    # deprecated parameters for compatibility
    parser.add_argument('--print_freq', type=int, default=10, help='DEPRECATED: use --log_interval instead')
    parser.add_argument('--save_freq', type=int, default=50, help='DEPRECATED: use --save_every instead')
    
    return parser

def load_vae_model(args, device):
    """Load pre-trained VAE model"""
    print(f"Loading VAE model from {args.vae_checkpoint}")
    
    if not os.path.exists(args.vae_checkpoint):
        raise FileNotFoundError(f"VAE checkpoint not found at {args.vae_checkpoint}")
    
    # Create VAE model
    vae = HyperspectralVAE(
        spectral_channels=args.input_channels,
        latent_channels=args.vae_latent_channels,
        base_channels=args.vae_base_channels,
    )
    
    # Load checkpoint
    ckpt = torch.load(args.vae_checkpoint, weights_only=False, map_location='cpu')
    vae.load_state_dict(ckpt['model_state_dict'], strict=True)
    
    for k, v in vae.named_parameters():
        if not torch.equal(v, ckpt['model_state_dict'][k]):
            print(f"Parameter {k} not loaded correctly.")
            raise ValueError(f"Parameter {k} not loaded correctly.")
        
    vae = vae.to(device)
    vae.eval()  # Set to evaluation mode since we're using it as a fixed encoder/decoder
    
    print(f"VAE model loaded successfully. Latent channels: {args.vae_latent_channels}")
    
    # Freeze VAE parameters
    for param in vae.parameters():
        param.requires_grad = False
    
    return vae

def main(args):
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"Using device: {device}")
    
    # create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # save arguments for reproducibility
    # import json
    # with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
    #     json.dump(vars(args), f, indent=2)
    
    # create dataset and dataloader
    train_dataset = HASCID_data(data_path=args.data_path, 
                                train_mode=args.train_mode, 
                                split='train', 
                                eval_ratio=args.eval_ratio, 
                                data_format=args.train_mode)
    eval_dataset = HASCID_data(data_path=args.data_path,
                                 train_mode=args.train_mode, 
                                 split='test', 
                                 eval_ratio=args.eval_ratio, 
                                 data_format=args.train_mode)
    
    if args.train_mode == 'pixel':
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, collate_fn=pixel_collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=args.num_workers, collate_fn=pixel_collate_fn)
    else:
        # Use image collate function for proper 2D image handling
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, collate_fn=image_collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=args.num_workers, collate_fn=image_collate_fn)
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of evaluation samples: {len(eval_dataset)}")
    
    # Load pre-trained VAE model
    vae = load_vae_model(args, device)
    
    # Create diffusion model - operates on VAE latent space
    model = LatentU2NetHyperspectral(
        sensor_channels=args.sensor_channels,
        latent_channels=args.vae_latent_channels,
        base_channels=args.diffusion_base_channels,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * args.lrf)
    scaler = torch.amp.GradScaler(enabled=(args.scaler == 'amp'))
    
    # Create diffusion trainer
    trainer = DiffusionTrainer(
        loss_type=args.loss_type,
        beta_schedule=args.noise_schedule,
    )
    
    # load trained diffusion model if exists
    start_epoch = 0
    if args.resume.endswith('.pth') and os.path.isfile(args.resume):
        print(f"Loading diffusion model from {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location='cpu')
        
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        vae.load_state_dict(ckpt['vae_state_dict'], strict=True)
        
        # Verify model loading
        for k, v in model.named_parameters():
            if not torch.equal(v, ckpt['model_state_dict'][k]):
                print(f"Parameter {k} not loaded correctly.")
                raise ValueError(f"Parameter {k} not loaded correctly.")
        print(f"Diffusion model loaded successfully from {args.resume}")
        for k, v in vae.named_parameters():
            if not torch.equal(v, ckpt['vae_state_dict'][k]):
                print(f"Parameter {k} not loaded correctly.")
                raise ValueError(f"Parameter {k} not loaded correctly.")
        print(f"VAE model loaded successfully from {args.resume}")
        
        if 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("Optimizer state loaded.")
        
        if 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
            lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print("LR scheduler state loaded.")
        elif 'lr_scheduler' in ckpt and ckpt['lr_scheduler'] is not None:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            print("LR scheduler state loaded.")
        
        start_epoch = ckpt['epoch'] + 1 if 'epoch' in ckpt else 0
        lr_scheduler.last_epoch = start_epoch - 1  # Adjust for lr_scheduler step
        print(f"Resuming training from epoch {start_epoch}")
        
    else:
        print("No diffusion model checkpoint found, training from scratch.")
    
    model = model.to(args.device)
    
    print(f"\nModel Architecture:")
    print(f"VAE: {args.input_channels} -> {args.vae_latent_channels} latent channels")
    print(f"Diffusion U-Net: {args.vae_latent_channels} input channels, {args.sensor_channels} condition channels")
    print(f"Total diffusion model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"VAE parameters (frozen): {sum(p.numel() for p in vae.parameters()):,}")
    
    # start training with the training functions from train_2d_with_vae.py
    trained_model, train_history = train_full_pipeline(
        model=model,
        vae=vae,
        diffusion_trainer=trainer,
        train_dataloader=train_loader,
        val_dataloader=eval_loader,
        num_epochs=args.num_epochs,
        device=args.device,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        save_dir=args.save_path,
        grad_clip=args.grad_clip,
        val_every=args.val_every,
        save_every=args.save_every,
        generate_every=args.generate_every,
        scaler=scaler
    )
    
    # save the training history as .txt file
    if train_history:
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


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    print("="*80)
    print("2D U-Net Diffusion Model with VAE Compression")
    print("="*80)
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*80)
    
    main(args)
