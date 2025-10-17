import torch
import misc.util as utils
from model.u2net_1d import *
from model.diffusion_trainer import *
import argparse
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from train_eval.train_1d import *
from data_loader.my_dataset import HASCID_data, pixel_collate_fn, image_collate_fn
from data_loader.HFD_dataset import HFD_data
import random


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='VAE for Hyperspectral Image Reconstruction')
    
    # dataset parameters
    parser.add_argument('--data_path', type=str, default='dataset/HFD100 Mat dataset', choices=['dataset/HASCID-Dataset',
                                                                                            'dataset/HFD100 Mat dataset'], help='path to dataset')
    parser.add_argument('--train_mode', type=str, default='pixel', choices=['pixel', 'image'], 
                        help='training mode, if pixel, then randomly sample pixels from all training images; \
                             if image, then randomly sample images')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to load data')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
    parser.add_argument('--eval_ratio', type=float, default=0.1, help='the ratio of test data during training')
    parser.add_argument('--R-n', type=int, default=1, help='the number of random measurements, if None, then use full measurements')
    parser.add_argument('--dataset', type=str, default='HASCID', choices=['HASCID', 'HFD'], help='dataset to use')
    
    # model parameters, VAE init input arguments
    # the diffusion conditioning is from the sensor response, input is the raw hsi channels
    parser.add_argument('--latent_channels', type=int, default=8, help='dimension of latent space of VAE compression')
    parser.add_argument('--sensor_channels', type=int, default=30, help='number of channels of the sensor response')
    parser.add_argument('--input_channels', type=int, default=128, help='number of spectral bands of input data')
    parser.add_argument('--base_channels', type=int, default=128, help='base channels of VAE model')
    
    # optimization parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: 10)')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency (default: 50)')
    parser.add_argument('--lrf', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--scaler', type=str, default='amp', choices=['none', 'amp'], help='use automatic mixed precision training')
    parser.add_argument('--loss-type', type=str, default='l1', choices=['l1', 'l2'], help='type of loss function')
    
    # output parameters
    parser.add_argument('--resume', type=str, default='results/1d_hsi_diffusion/final_model', help='path to resume a checkpoint')
    parser.add_argument('--save_path', type=str, default='results/1d_hsi_diffusion/HFD/R_1/l1_loss', help='path to save results')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:1', help='device to use for computation')
    
    return parser

def main(args):
    # save the args as a text file
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    import json
    with open(os.path.join(args.save_path, 'args_1d.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
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
    
    if args.train_mode == 'pixel':
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, collate_fn=pixel_collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=args.num_workers, collate_fn=pixel_collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=args.num_workers)
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of evaluation samples: {len(eval_dataset)}")
    
    # create model, the diffusion 1d model, no VAE is used in 1d diffusion model
    model = U2Net1D(
        input_channels=1,
        condition_dim=args.sensor_channels,
        base_channels=args.base_channels,
    )
    print(f"Model created: U2Net1D with base channels {args.base_channels}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * args.lrf)
    scaler = torch.amp.GradScaler(enabled=(args.scaler == 'amp'))
     
    # load trained model if exists
    if args.resume.endswith('.pth') and os.path.isfile(args.resume):
        print(f"Loading model from {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location='cpu')
        
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        
        for k, v in model.named_parameters():
            if not torch.equal(v, ckpt['model_state_dict'][k]):
                print(f"Parameter {k} not loaded correctly.")
                # raise ValueError(f"Parameter {k} not loaded correctly.")
                raise ValueError(f"Parameter {k} not loaded correctly.")
        print(f"Model loaded successfully from {args.resume}")
        
        if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            print("Optimizer state loaded.")
        
        if 'lr_scheduler' in ckpt and ckpt['lr_scheduler'] is not None:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            print("LR scheduler state loaded.")
        
        start_epoch = ckpt['epoch'] + 1 if 'epoch' in ckpt else 0
        # lr_scheduler.last_epoch = start_epoch - 1  # Adjust for lr_scheduler step
        print(f"Resuming training from epoch {start_epoch}")
        
    else:
        print("No checkpoint found, training from scratch.")
    
        start_epoch = 0
    
    model = model.to(args.device)
    
    trainer = DiffusionTrainer(
        loss_type=args.loss_type,
        device=args.device,
    )
    
    # start training with the training and eval functions from train_1d.py
    best_model, train_history = train_full_pipeline(
        model=model,
        diffusion_trainer=trainer,
        train_dataloader=train_loader,
        val_dataloader=eval_loader,
        num_epochs=args.num_epochs,
        device=args.device,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        save_dir=args.save_path,
        scaler=scaler,
        save_every=args.save_freq,
        log_interval=args.print_freq,
    )
    
    # save the training history as .txt file
    # Convert list of dictionaries to a more suitable format for saving
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
            
    print("="*80)
    print("1D HSI Diffusion Model for Direct Hyperspectral Reconstruction")
    print("="*80)
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*80)
    
    main(args)