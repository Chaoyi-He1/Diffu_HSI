import torch.dataset as Dataset
import os
import numpy as np
import random
import h5py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F


class HASCID_data(Dataset.Dataset):
    def __init__(self, data_path, train_mode='pixel', eval_ratio=0.1, split='train'):
        super(HASCID_data, self).__init__()
        pass
        self.data_path = data_path
        self.train_mode = train_mode
        self.eval_ratio = eval_ratio
        self.split = split  # 'train' or 'test'
        
        assert self.train_mode in ['pixel', 'image'], "train_mode must be 'pixel' or 'image'"
        assert self.split in ['train', 'test'], "split must be 'train' or 'test'"
        assert os.path.exists(self.data_path), f"Data path {self.data_path} does not exist"
        
        # Load data path
        self.scene_path = os.path.join(self.data_path, 'rw')
        self.gt_path = os.path.join(self.data_path, 'gt_files')
        
        self.img_list = sorted([f for f in os.listdir(self.scene_path) if f.endswith('.npy')])
        self.img_name = [os.path.splitext(f)[0][3:] for f in self.img_list] # remove 'rw_' prefix and '.npy' suffix
    
    def split_data(self):
        if self.split == 'train':
            self.img_list = self.img_list[:int(len(self.img_list) * (1 - self.eval_ratio))]
            self.img_name = self.img_name[:int(len(self.img_name) * (1 - self.eval_ratio))]
        else:
            self.img_list = self.img_list[int(len(self.img_list) * (1 - self.eval_ratio)):]
            self.img_name = self.img_name[int(len(self.img_name) * (1 - self.eval_ratio)):]
            
    def load_data(self, idx):
        scene_name = self.img_name[idx]
        gt_name
        