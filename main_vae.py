import torch
import misc.util as utils
from model.hyperspectral_vae import *
import argparse
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from train_eval.train_vae import *


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='VAE for Hyperspectral Image Reconstruction')
    
    # dataset parameters
    parser.add_argument('--data_path', type=str, default='dataset/HASCID-Dataset', help='path to dataset')
    parser.add_argument('--train_mode', type=str, default='pixel', choices=['pixel', 'image'], 
                        help='training mode, if pixel, then randomly sample pixels from all training images; \
                             if image, then randomly sample images')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to load data')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training')
    parser.add_argument('--eval_ratio', type=float, default=0.1, help='the ratio of test data during training')
    
    # VAE model parameters, VAE init input arguments
    parser.add_argument('--latent_channels', type=int, default=8, help='dimension of latent space of VAE compression')
    parser.add_argument('--input_channels', type=int, default=128, help='number of spectral bands of input data')
    parser.add_argument('--base_channels', type=int, default=128, help='base channels of VAE model')
    
    # optimization parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: 10)')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency (default: 50)')
    parser.add_argument('--lrf', type=float, default=0.1, help='learning rate decay factor')
    
    # output parameters
    parser.add_argument('--save_path', type=str, default='results/vae_hyperspectral', help='path to save results')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for computation')
    
    return parser

def main(args):
    # set random seed
    utils.set_random_seed(args.seed)
    
    # create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    