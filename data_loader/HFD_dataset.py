import torch.utils.data as Dataset
import os
import numpy as np
import random
import scipy.io as sio
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from scipy.interpolate import interp1d


class HFD_data(Dataset.Dataset):
    def __init__(self, data_path, train_mode='pixel', eval_ratio=0.1, split='train', data_format='pixel', type='Flower', R_n=None):
        super(HFD_data, self).__init__()
        self.data_path = data_path
        self.train_mode = train_mode  # 'pixel' or 'image'
        self.eval_ratio = eval_ratio
        self.split = split  # 'train' or 'test'
        self.data_format = data_format
        self.pixel_num = 16  # number of pixels to sample if data_format is 'pixel'
        self.type = type  # 'Flower' or 'Vegetable'
        self.use_new_R = False if R_n is None else True
        self.R_n = R_n  # if use_new_R is True, load sensor response from R_Device{R_n}.mat
        
        assert self.train_mode in ['pixel', 'image'], "train_mode must be 'pixel' or 'image'"
        assert self.split in ['train', 'test'], "split must be 'train' or 'test'"
        assert self.data_format in ['pixel', 'image'], "data_format must be 'pixel' or 'image'"
        assert self.type in ['Flower', 'Leaves', 'Scenses'], "type must be 'Flower' or 'Leaves' or 'Scenses'"
        
        # from 451 to 855, total 31 bands
        self.wavelens = np.linspace(451, 855, 31)
        new_wavelengths = np.linspace(self.wavelens[0], self.wavelens[-1], 64)
        self.new_wavelens = new_wavelengths

        # paths
        self.data_folder = os.path.join(self.data_path, f'Mat{self.type}60', 'Train')
        # get all the subfolders's .mat files in self.data_folder
        self.img_list = []
        for folder in os.listdir(self.data_folder):
            folder_path = os.path.join(self.data_folder, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.mat'):
                        self.img_list.append(os.path.join(folder_path, file))
        self.img_list.sort()  # sort the list to ensure the order is the same every time

        self.split_data()  # split the data into train and test sets
        
        if self.use_new_R:
            print(f"Loading sensor response from R_Device{self.R_n}.mat")
            self.load_sensor_response_new(self.R_n)
        else:
            print(f"Loading sensor response from PH5_interp_results.mat")
            self.load_sensor_response()
        
    def load_sensor_response(self):
        # Load sensor response matrix from .mat file
        sensor_response_path = os.path.join(self.data_path, 'PH5_interp_results.mat')
        sensor_R_matrix = sio.loadmat(sensor_response_path)['PH5_interp']    # [C, N], where C is the number of channels, N is the number of sensor responses
        self.sensor_R_matrix = np.array(sensor_R_matrix)  # [C, N]
        
        # use OSP method to select 30 channels from the sensor response
        # self.sensor_R_matrix, selected_indices = osp(self.sensor_R_matrix, num_channels=30)
       
        # Uniformly select 30 channels from the sensor response
        num_total_channels = self.sensor_R_matrix.shape[1]
        selected_indices = np.linspace(0, num_total_channels - 1, 30, dtype=int).tolist()
        self.sensor_R_matrix = self.sensor_R_matrix[:, selected_indices]
        
        self.selected_indices = selected_indices
        
        self.sensor_wavelens = np.arange(400, 1560, 10)  # Example wavelengths from 400nm to 1550nm with a step of 10nm

        # Resample the sensor response to self.wavelens
        f = interp1d(self.sensor_wavelens, self.sensor_R_matrix, axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
        self.sensor_R_matrix = f(self.wavelens)
        assert self.sensor_R_matrix.shape[0] == len(self.wavelens), f"Sensor response shape {self.sensor_R_matrix.shape[0]} does not match wavelength length {len(self.wavelens)}"
        
        # Normalize the sensor response matrix to [0, 1] for each column
        R_min, R_max = self.sensor_R_matrix.min(axis=0), self.sensor_R_matrix.max(axis=0)
        self.sensor_R_matrix = (self.sensor_R_matrix - R_min) / (R_max - R_min + 1e-20)
    
    def load_sensor_response_new(self, n):
        '''
        Load sensor response not from PH5_interp_results.mat, but from R_Device1 or R_Device2.mat files
        '''
        sensor_response_path = os.path.join(self.data_path, f'R_Device{n}.mat')
        sensor_R_matrix = sio.loadmat(sensor_response_path)['R']    # [C, N], where C is the number of channels, N is the number of sensor responses
        self.sensor_R_matrix = np.array(sensor_R_matrix)  # [C, N]
        
        # use OSP method to select 30 channels from the sensor response
        # self.sensor_R_matrix, selected_indices = osp(self.sensor_R_matrix, num_channels=30)
       
        # Uniformly select 30 channels from the sensor response
        num_total_channels = self.sensor_R_matrix.shape[1]
        selected_indices = np.linspace(0, num_total_channels - 1, 30, dtype=int).tolist()
        self.sensor_R_matrix = self.sensor_R_matrix[:, selected_indices]
        
        self.selected_indices = selected_indices

        # sensor wavelengths first wave length is 400, last wave length is 1000
        self.sensor_wavelens = np.linspace(400, 1000, self.sensor_R_matrix.shape[0]) 
        
        # Resample the sensor response to self.wavelens
        f = interp1d(self.sensor_wavelens, self.sensor_R_matrix, axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
        self.sensor_R_matrix = f(self.wavelens)
        assert self.sensor_R_matrix.shape[0] == len(self.wavelens), f"Sensor response shape {self.sensor_R_matrix.shape[0]} does not match wavelength length {len(self.wavelens)}"

        # Normalize the sensor response matrix to [0, 1] for each column
        R_min, R_max = self.sensor_R_matrix.min(axis=0), self.sensor_R_matrix.max(axis=0)
        self.sensor_R_matrix = (self.sensor_R_matrix - R_min) / (R_max - R_min + 1e-20)
        
        print(f"Sensor response min {self.sensor_R_matrix.min()}, max {self.sensor_R_matrix.max()}")
        
    
    def expand_wavelens(self, data):
        '''
        Expand the wavelength dimension of data from 31 bands to 64 bands by interpolation
        Input:
            data: [H, W, 31]
        Output:
            data_expanded: [H, W, 64]
        '''
        # Create a new wavelength axis with 64 points
        new_wavelengths = np.linspace(self.wavelens[0], self.wavelens[-1], 64)
        # self.new_wavelens = new_wavelengths
        
        # Interpolate the data to the new wavelength axis
        data_expanded = np.zeros((data.shape[0], data.shape[1], 64))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                f = interp1d(self.wavelens, data[i, j], kind='linear', bounds_error=False, fill_value="extrapolate")
                data_expanded[i, j] = f(new_wavelengths)

        return data_expanded
    
    def split_data(self):
        # Split the img_list into train and test sets based on eval_ratio
        num_total = len(self.img_list)
        num_eval = int(num_total * self.eval_ratio)
        num_train = num_total - num_eval
        
        if self.split == 'train':
            self.img_list = self.img_list[:num_train]
        else:
            self.img_list = self.img_list[num_train:]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # load the .mat file
        gt_data = sio.loadmat(self.img_list[idx])['truth']  # [H, W, 31]
        gt_data = np.array(gt_data, dtype=np.float32)
        
        # normalize gt_data to [-1, 1]
        gt_min, gt_max = gt_data.min(), gt_data.max()
        gt_data = (gt_data - gt_min) / (gt_max - gt_min + 1e-20)
        gt_data = gt_data * 2.0 - 1.0
        
        # calculate sensor response based on self.sensor_R_matrix
        H, W, C = gt_data.shape
        sensor_data = np.matmul(gt_data.reshape(-1, C), self.sensor_R_matrix)  # [H*W, N]
        sensor_data = sensor_data.reshape(H, W, -1)  # [H, W, N]
        
        # expand gt_data to 64 bands
        gt_data = self.expand_wavelens(gt_data)  # [H, W, 64]
        
        if self.data_format == 'pixel':
            # randomly sample self.pixel_num pixels from the image
            pixel_indices = random.sample(range(H * W), self.pixel_num)
            gt_pixels = gt_data.reshape(-1, gt_data.shape[2])[pixel_indices, :]  # [self.pixel_num, 64]
            sensor_pixels = sensor_data.reshape(-1, sensor_data.shape[2])[pixel_indices, :]  # [self.pixel_num, N]
            return gt_pixels, sensor_pixels
        elif self.data_format == 'image':
            return gt_data, sensor_data

@staticmethod
def pixel_collate_fn(batch):
    gt_data, sensor_data = list(zip(*batch))
    # the pixel data format is [B, pixel_num, C], 
    # where B is the batch size, pixel_num is the number of pixels sampled from each image, C is the number of channels
    # we need to merge the first two dimensions to get [B*pixel_num, C]
    gt_data = torch.tensor(np.concatenate(gt_data, axis=0), dtype=torch.float32)  # [B, pixel_num, C']
    sensor_data = torch.tensor(np.concatenate(sensor_data, axis=0), dtype=torch.float32)  # [B, pixel_num, N]

    gt_data = gt_data.reshape(-1, 1, gt_data.shape[-1])  # [B*pixel_num, 1, C']
    sensor_data = sensor_data.reshape(-1, sensor_data.shape[-1])  # [B*pixel_num, N]

    return gt_data, sensor_data

@staticmethod
def image_collate_fn(batch):
    gt_data, sensor_data = list(zip(*batch))
    # the image data format is [H, W, C] from dataset
    # we need to convert to [B, C, H, W] for VAE training
    gt_data = torch.tensor(np.stack(gt_data, axis=0), dtype=torch.float32)  # [B, H, W, C']
    sensor_data = torch.tensor(np.stack(sensor_data, axis=0), dtype=torch.float32)  # [B, H, W, N]
    
    # Convert from [B, H, W, C] to [B, C, H, W]
    gt_data = gt_data.permute(0, 3, 1, 2)  # [B, C', H, W]
    sensor_data = sensor_data.permute(0, 3, 1, 2)  # [B, N, H, W]
    
    return gt_data, sensor_data


if __name__ == '__main__':
    # check the min max value of the sensor_data and gt_data
    dataset = HFD_data(data_path='dataset/HFD100 Mat dataset', train_mode='image', eval_ratio=0.1, split='train', data_format='image', type='Flower', R_n=1)
    print(f"Number of samples in the dataset: {len(dataset)}")
    for i in range(len(dataset)):
        gt_data, sensor_data = dataset[i]
        print(f"Sample {i}: gt_data min {gt_data.min()}, max {gt_data.max()}; sensor_data min {sensor_data.min()}, max {sensor_data.max()}")
        if i == 10:
            break   