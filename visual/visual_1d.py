import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from torch.utils.data import DataLoader
import sys
import random
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader.my_dataset import HASCID_data, pixel_collate_fn
from data_loader.HFD_dataset import HFD_data
from model.u2net_1d import U2Net1D
from model.diffusion_trainer import DiffusionTrainer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Visualization for 1D HSI Diffusion Model')
    
    # dataset parameters
    parser.add_argument('--data_path', type=str, default='dataset/HFD100 Mat dataset', choices=['dataset/HASCID-Dataset',
                                                                                            'dataset/HFD100 Mat dataset'], help='path to dataset')
    parser.add_argument('--train_mode', type=str, default='pixel', choices=['pixel', 'image'], 
                        help='training mode')
    parser.add_argument('--eval_ratio', type=float, default=0.1, help='the ratio of test data during training')
    parser.add_argument('--R-n', type=int, default=1, help='the number of random measurements, if None, then use full measurements')
    parser.add_argument('--dataset', type=str, choices=['HASCID', 'HFD'], default='HFD', help='which dataset to use')
    
    # model parameters
    parser.add_argument('--sensor_channels', type=int, default=30, help='number of channels of the sensor response')
    parser.add_argument('--input_channels', type=int, default=128, help='number of spectral bands of input data')
    parser.add_argument('--base_channels', type=int, default=128, help='base channels of model')
    
    # visualization parameters
    parser.add_argument('--num_examples', type=int, default=5, help='number of examples to visualize')
    parser.add_argument('--model_path', type=str, default='results/1d_hsi_diffusion/HFD/R_1/checkpoint_epoch_201.pth', help='path to trained model checkpoint')
    parser.add_argument('--save_path', type=str, default='results/1d_visualization', help='path to save visualization results')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='dataset split to visualize')
    
    # generation parameters
    parser.add_argument('--num_steps', type=int, default=1000, help='number of denoising steps')
    parser.add_argument('--method', type=str, default='ddpm', choices=['ddpm', 'ddim'], help='sampling method')
    
    # device
    parser.add_argument('--device', type=str, default='cuda:1', help='device to use for computation')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    return parser


def load_model(model_path: str, device: torch.device, sensor_channels: int = 30, base_channels: int = 128):
    """Load the trained 1D diffusion model"""
    model = U2Net1D(
        input_channels=1,
        condition_dim=sensor_channels,
        base_channels=base_channels,
    )
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    return model


def generate_samples(model, diffusion_trainer, sensor_data, device, num_steps=50, method='ddpm'):
    """Generate samples using the trained diffusion model"""
    model.eval()
    
    with torch.no_grad():
        # sensor_data shape: [batch_size, sensor_channels] 
        batch_size = sensor_data.shape[0]
        sequence_length = 160 if args.dataset == 'HASCID' else 64 # Based on the hyperspectral data dimension from dataset
        
        # Generate samples
        generated = diffusion_trainer.sample(
            model=model,
            cond=sensor_data,
            shape=(batch_size, 1, sequence_length),
            n_steps=num_steps,
            method=method,
            progress=True
        )
        
    return generated


def create_comparison_plot(gt_data, sensor_data, generated_data, wavelengths, sensor_wavelengths, 
                         example_idx, save_path):
    """Create comparison plots for a single example"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Example {example_idx + 1}: HSI Reconstruction Comparison', fontsize=16)
    
    # Plot 1: Ground Truth Hyperspectral Data
    axes[0, 0].plot(wavelengths, gt_data.squeeze(), 'b-', linewidth=2, label='Ground Truth')
    axes[0, 0].set_title('Ground Truth Hyperspectral Data')
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Reflectance')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Sensor Response Data (Conditioning)
    axes[0, 1].plot(sensor_wavelengths, sensor_data.squeeze(), 'r-', linewidth=2, 
                   marker='o', markersize=4, label='Sensor Response')
    axes[0, 1].set_title('Sensor Response Data (Conditioning)')
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Response')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Generated Hyperspectral Data
    axes[1, 0].plot(wavelengths, generated_data.squeeze(), 'g-', linewidth=2, label='Generated')
    axes[1, 0].set_title('Generated Hyperspectral Data')
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Reflectance')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Overlay Comparison
    axes[1, 1].plot(wavelengths, gt_data.squeeze(), 'b-', linewidth=2, 
                   label='Ground Truth', alpha=0.8)
    axes[1, 1].plot(wavelengths, generated_data.squeeze(), 'g--', linewidth=2, 
                   label='Generated', alpha=0.8)
    axes[1, 1].set_title('Ground Truth vs Generated')
    axes[1, 1].set_xlabel('Wavelength (nm)')
    axes[1, 1].set_ylabel('Reflectance')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Calculate and display metrics
    mse = np.mean((gt_data.squeeze() - generated_data.squeeze()) ** 2)
    mae = np.mean(np.abs(gt_data.squeeze() - generated_data.squeeze()))
    
    # Add metrics text
    metrics_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}'
    axes[1, 1].text(0.02, 0.98, metrics_text, transform=axes[1, 1].transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save individual plot
    plot_path = os.path.join(save_path, f'comparison_example_{example_idx + 1}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mse, mae


def create_summary_plot(all_metrics, save_path):
    """Create a summary plot showing metrics across all examples"""
    
    mse_values = [m['mse'] for m in all_metrics]
    mae_values = [m['mae'] for m in all_metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE plot
    ax1.bar(range(1, len(mse_values) + 1), mse_values, color='skyblue', alpha=0.7)
    ax1.set_title('Mean Squared Error (MSE) per Example')
    ax1.set_xlabel('Example Number')
    ax1.set_ylabel('MSE')
    ax1.grid(True, alpha=0.3)
    
    # MAE plot
    ax2.bar(range(1, len(mae_values) + 1), mae_values, color='lightcoral', alpha=0.7)
    ax2.set_title('Mean Absolute Error (MAE) per Example')
    ax2.set_xlabel('Example Number')
    ax2.set_ylabel('MAE')
    ax2.grid(True, alpha=0.3)
    
    # Add mean lines
    ax1.axhline(y=np.mean(mse_values), color='red', linestyle='--', 
               label=f'Average MSE: {np.mean(mse_values):.6f}')
    ax2.axhline(y=np.mean(mae_values), color='red', linestyle='--', 
               label=f'Average MAE: {np.mean(mae_values):.6f}')
    
    ax1.legend()
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'summary_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Average MSE: {np.mean(mse_values):.6f} ± {np.std(mse_values):.6f}")
    print(f"Average MAE: {np.mean(mae_values):.6f} ± {np.std(mae_values):.6f}")
    print(f"Min MSE: {np.min(mse_values):.6f}")
    print(f"Max MSE: {np.max(mse_values):.6f}")
    print(f"Min MAE: {np.min(mae_values):.6f}")
    print(f"Max MAE: {np.max(mae_values):.6f}")


def visualize_results(args):
    """Main visualization function"""
    
    # Set up device and random seed
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    if args.dataset == 'HASCID':
        dataset = HASCID_data(
            data_path=args.data_path,
            train_mode=args.train_mode,
            split=args.split,
            eval_ratio=args.eval_ratio,
            data_format=args.train_mode,
            R_n=args.R_n,
        )
    elif args.dataset == 'HFD':
        dataset = HFD_data(
            data_path=args.data_path,
            train_mode=args.train_mode,
            split=args.split,
            eval_ratio=args.eval_ratio,
            data_format=args.train_mode,
            R_n=args.R_n,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Create dataloader
    if args.train_mode == 'pixel':
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, 
                              collate_fn=pixel_collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # Load model
    model = load_model(args.model_path, device, args.sensor_channels, args.base_channels)
    model.to(device)
    
    # Initialize diffusion trainer
    diffusion_trainer = DiffusionTrainer(device=device)
    
    # Get wavelengths from dataset
    # GT data uses first 160 wavelengths, sensor data corresponds to the clipped sensor wavelengths
    wavelengths = dataset.new_wavelens if args.dataset == 'HFD' else dataset.wavelens[:160]  # Match GT data dimension
    
    # For sensor data, we need to create appropriate x-axis values
    # The sensor data has 30 channels corresponding to the selected wavelength range
    sensor_data_sample = next(iter(dataloader))[1]  # Get sensor data shape
    sensor_channels = sensor_data_sample.shape[-1]  # Should be 30
    
    # Create wavelength array for sensor data (uniformly spaced in [400, 1000] range)
    sensor_wavelengths = np.linspace(400, 1000, sensor_channels)
    
    print(f"Generating visualizations for {args.num_examples} examples...")
    
    all_metrics = []
    examples_processed = 0
    
    # Generate and visualize examples
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing examples")):
        if examples_processed >= args.num_examples:
            break
        if  batch_idx <= 1:
            continue
        # Extract data from batch
        gt_data, sensor_data = batch
        
        # Move to device
        gt_data = gt_data.to(device)
        sensor_data = sensor_data.to(device)
        
        # For pixel mode, we get batches of pixels. Let's visualize each pixel separately
        batch_size = gt_data.shape[0]
        
        for pixel_idx in range(batch_size):
            if examples_processed >= args.num_examples:
                break
                
            # Extract single pixel data
            single_gt = gt_data[pixel_idx:pixel_idx+1]  # [1, channels]
            single_sensor = sensor_data[pixel_idx:pixel_idx+1]  # [1, sensor_channels]
            
            # Reshape for 1D model: [batch, channels] -> [batch, 1, channels]  
            if single_gt.dim() == 2:
                single_gt_1d = single_gt.unsqueeze(1)  # [1, 1, channels]
            else:
                single_gt_1d = single_gt

            # Generate samples
            with torch.no_grad():
                generated_data = generate_samples(
                    model, diffusion_trainer, single_sensor, device, 
                    num_steps=args.num_steps, method=args.method
                )
            
            # Move back to CPU for plotting
            gt_cpu = single_gt.cpu().numpy()
            sensor_cpu = single_sensor.cpu().numpy()
            generated_cpu = generated_data.squeeze(1).cpu().numpy()  # Remove the channel dimension
            
            # Create comparison plot for this pixel
            mse, mae = create_comparison_plot(
                gt_cpu, sensor_cpu, generated_cpu,
                wavelengths, sensor_wavelengths,
                examples_processed, args.save_path
            )
            
            all_metrics.append({'mse': mse, 'mae': mae})
            examples_processed += 1
        

        
    # Create summary plot
    create_summary_plot(all_metrics, args.save_path)
    
    # Save metrics to file
    metrics_file = os.path.join(args.save_path, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Example\tMSE\tMAE\n")
        for i, metrics in enumerate(all_metrics):
            f.write(f"{i+1}\t{metrics['mse']:.6f}\t{metrics['mae']:.6f}\n")
        f.write(f"\nAverage\t{np.mean([m['mse'] for m in all_metrics]):.6f}\t{np.mean([m['mae'] for m in all_metrics]):.6f}\n")
    
    print(f"\nVisualization complete! Results saved to: {args.save_path}")
    print(f"- Individual comparison plots: comparison_example_*.png")
    print(f"- Summary metrics plot: summary_metrics.png")
    print(f"- Detailed metrics: metrics.txt")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    visualize_results(args)
