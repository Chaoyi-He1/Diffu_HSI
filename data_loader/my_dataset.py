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
        
        