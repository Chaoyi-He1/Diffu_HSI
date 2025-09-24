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


class HASCID_data(Dataset.Dataset):
    def __init__(self, data_path, train_mode='pixel', eval_ratio=0.1, split='train', data_format='pixel'):
        super(HASCID_data, self).__init__()
        pass
        self.data_path = data_path
        self.train_mode = train_mode
        self.eval_ratio = eval_ratio
        self.split = split  # 'train' or 'test'
        self.data_format = data_format
        self.pixel_num = 16  # number of pixels to sample if data_format is 'pixel'
        
        assert self.train_mode in ['pixel', 'image'], "train_mode must be 'pixel' or 'image'"
        assert self.split in ['train', 'test'], "split must be 'train' or 'test'"
        assert self.data_format in ['pixel', 'image'], "data_format must be 'pixel' or 'image'"
        assert os.path.exists(self.data_path), f"Data path {self.data_path} does not exist"
        
        # Load data path
        self.scene_path = os.path.join(self.data_path, 'rw')
        self.gt_path = os.path.join(self.data_path, 'gt_files')
        self.resample_path = os.path.join(self.data_path, 'resampled_gt')
        
        self.img_list = sorted([f for f in os.listdir(self.scene_path) if f.endswith('.npy')])
        self.img_name = [os.path.splitext(f)[0][3:] for f in self.img_list] # remove 'rw_' prefix and '.npy' suffix

        self.wavelens = np.array([397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29, 426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25, 455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32, 484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48, 513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75, 542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12, 572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60, 601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18, 631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87, 660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65, 690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54, 720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54, 750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64, 780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84, 810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14, 841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55, 871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06, 902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68, 932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40, 963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22, 994.31, 997.40, 1000.49, 1003.58])

        self.load_sensor_response()
        
        # if readsampled data does not exist, create it by resampling
        if not os.path.exists(self.resample_path) or len(os.listdir(self.resample_path)) != len(os.listdir(self.gt_path)):
            self.resample()
        
        self.split_data()
        del self.img_list
        
        # load the min and max values of the resampled data and sensor data
        data_range_file = os.path.join(self.data_path, 'data_range.npz')
        if os.path.exists(data_range_file):
            data_range = np.load(data_range_file)
            self.min_gt = data_range['min_gt']
            self.max_gt = data_range['max_gt']
            self.min_sensor = data_range['min_sensor']
            self.max_sensor = data_range['max_sensor']
        else:
            self.min_gt, self.max_gt, self.min_sensor, self.max_sensor = 0, 1, 0, 1
    
    def __len__(self):
        return len(self.img_name)
            
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

        # clip the sensor response to the range of [400, 1000] nm
        valid_indices = np.where((self.sensor_wavelens >= 400) & (self.sensor_wavelens <= 1000))[0]
        self.sensor_R_matrix = self.sensor_R_matrix[valid_indices, :]
        self.sensor_wavelens = self.sensor_wavelens[valid_indices]
        
        # Normalize the sensor response matrix to [0, 1]
        R_min, R_max = self.sensor_R_matrix.min(), self.sensor_R_matrix.max()
        self.sensor_R_matrix = (self.sensor_R_matrix - R_min) / (R_max - R_min)
    
    def resample(self):
        from tqdm import tqdm
        print("Resampling hyperspectral data to match sensor response wavelengths...")
        assert len(os.listdir(self.gt_path)) == len(self.img_list), "Number of ground truth files does not match number of scene files"
        # Resample the hyperspectral data to match the sensor response wavelengths and save the resampled data to a new folder as .npy files
        resample_path = os.path.join(self.data_path, 'resampled_gt')
        if not os.path.exists(resample_path):
            os.makedirs(resample_path)

        min_gt, max_gt, min_sensor, max_sensor = 0, 0, 0, 0
        tqbar = tqdm(self.img_name, desc='Resampling', ncols=100)
        for i, img_name in enumerate(tqbar):
            # print img_name in the tqdbar
            tqbar.set_postfix({"img_name": img_name})
            
            if os.path.exists(os.path.join(resample_path, f'resamp_gt_{img_name}.npy')):
                continue
            
            gt_file = os.path.join(self.gt_path, f'gtRef_{img_name}.npy')
            gt_data = np.load(gt_file)  # [H, W, C]
            assert gt_data.shape[2] == len(self.wavelens), f"Ground truth data channel {gt_data.shape[2]} does not match wavelength length {len(self.wavelens)}"
            H, W, C = gt_data.shape
            gt_data = gt_data.reshape(-1, C)  # [H*W, C]
            
            # interpolate the gt_data to the sensor_wavelens use scipy
            resampled_data = np.zeros((H * W, len(self.sensor_wavelens)), dtype=np.float32)
            for j in range(H * W):
                f = interp1d(self.wavelens, gt_data[j, :], kind='linear', bounds_error=False, fill_value="extrapolate")
                resampled_data[j, :] = f(self.sensor_wavelens)
            resampled_data = resampled_data.reshape(H, W, len(self.sensor_wavelens))  # [H, W, len(sensor_wavelens)]
            np.save(os.path.join(resample_path, f'resamp_gt_{img_name}.npy'), resampled_data)
            
            # update min and max values
            min_gt = min(min_gt, resampled_data.min())
            max_gt = max(max_gt, resampled_data.max())
            sensor_data = np.matmul(resampled_data.reshape(-1, resampled_data.shape[2]), self.sensor_R_matrix)  # [H*W, N]
            min_sensor = min(min_sensor, sensor_data.min())
            max_sensor = max(max_sensor, sensor_data.max())
            
        print(f"Resampling completed. Resampled data saved to {resample_path}")
        print(f"Resampled GT data range: min {min_gt}, max {max_gt}")
        print(f"Sensor data range after resampling: min {min_sensor}, max {max_sensor}")
        # save the min and max values to a npz file
        np.savez(os.path.join(self.data_path, 'data_range.npz'), min_gt=min_gt, max_gt=max_gt, min_sensor=min_sensor, max_sensor=max_sensor)
            
    def split_data(self):
        if self.split == 'train':
            self.img_list = self.img_list[:int(len(self.img_list) * (1 - self.eval_ratio))]
            self.img_name = self.img_name[:int(len(self.img_name) * (1 - self.eval_ratio))]
        else:
            self.img_list = self.img_list[int(len(self.img_list) * (1 - self.eval_ratio)):]
            self.img_name = self.img_name[int(len(self.img_name) * (1 - self.eval_ratio)):]
            
    def __getitem__(self, idx):
        scene_name = self.img_name[idx]
        resamp_file = os.path.join(self.resample_path, f'resamp_gt_{scene_name}.npy')
        gt_file = os.path.join(self.gt_path, f'gtRef_{scene_name}.npy')
        
        resamp_data = np.load(resamp_file)  # [H, W, C], where C is the sensor response channels
        gt_data = np.load(gt_file)  # [H, W, C'], original ground truth data, 204 channels in HASCID dataset

        gt_data = gt_data[:, :, :192]  # use the first 192 channels as ground truth
        
        # Rescale gt_data to [-1, 1], original range is [0, 1]
        gt_data = (gt_data - 0.5) * 2.0

        # use the resampled data to calculate the sensor data by multiplying with the sensor response matrix
        # reshape resamp_data to [H*W, C] and sensor_R_matrix to [C, N], then do matrix multiplication to get [H*W, N] 
        H, W, C = resamp_data.shape
        resamp_data_reshaped = resamp_data.reshape(-1, C)  # [H*W, C]
        sensor_data = np.matmul(resamp_data_reshaped, self.sensor_R_matrix)  # [H*W, N]
        sensor_data = sensor_data.reshape(H, W, -1)  # [H, W, N]
        sensor_data = (sensor_data - self.min_sensor) / (self.max_sensor - self.min_sensor)  # normalize to [0, 1]
        
        if self.data_format == 'pixel':
            # randomly sample 10 pixels from the image
            pixel_indices = random.sample(range(H * W), self.pixel_num)
            resamp_data = resamp_data_reshaped[pixel_indices, :]  # [pixel_num, C]
            sensor_data = sensor_data.reshape(-1, sensor_data.shape[2])[pixel_indices, :]  # [pixel_num, N]
            gt_data = gt_data.reshape(-1, gt_data.shape[2])[pixel_indices, :]  # [pixel_num, C']
            return gt_data, sensor_data
        elif self.data_format == 'image':
            # return the whole image
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

# OSP algorithm implementation
def osp(X, num_channels):
    ''' Orthogonal Subspace Projection (OSP) algorithm to 
    select a subset of channels from the sensor response matrix X.
    Args:
        X: Sensor response matrix of shape (C, N), where C is the number of channels, N is the number of sensor responses
        num_channels: Number of channels to select
        
    Returns:
        Selected channels from the sensor response matrix and their indices
    '''
    C, N = X.shape
    
    X_min, X_max = X.min(), X.max()
    X = (X - X_min) / (X_max - X_min)  # Normalize to [0, 1]
    
    selected_indices = []
    residual = X.copy().astype(np.float64)  # Use float64 for better numerical stability
    
    for _ in range(num_channels):
        # Compute the norms of each sensor response in the residual
        norms = np.linalg.norm(residual, axis=0)  # [N]
        selected_index = np.argmax(norms)
        selected_indices.append(selected_index)
        
        # Get the selected sensor response
        selected_response = residual[:, selected_index].reshape(-1, 1)  # [C, 1]
        
        # Avoid division by zero - compute squared norm (||v||²)
        norm_squared = np.linalg.norm(selected_response) ** 2  # Most explicit for OSP
        if norm_squared < 1e-12:
            break
            
        # Project out the selected response from all remaining columns
        # P = v * v^T / (v^T * v) where v is the selected response
        projection_matrix = (selected_response @ selected_response.T) / norm_squared  # [C, C]
        
        # Apply orthogonal projection: residual = (I - P) * residual
        residual = residual - projection_matrix @ residual
        
        # Set the selected column to zero to avoid selecting it again
        residual[:, selected_index] = 0
    
    # return the selected channels
    return X[:, selected_indices], selected_indices
        
        
if __name__ == '__main__':
    data_path = '/data/chaoyi_he/HSI/Diffu/dataset/HASCID-Dataset'
    dataset = HASCID_data(data_path, train_mode='image', split='train')