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
from data_loader.my_dataset import HASCID_data, image_collate_fn
from data_loader.HFD_dataset import HFD_data
from model.u2net_hyperspectral import U2NetHyperspectral
from model.diffusion_trainer import DiffusionTrainer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Visualization for 2D HSI Diffusion Model')
    
    # dataset parameters
    parser.add_argument('--data_path', type=str, default='dataset/HASCID-Dataset', choices=['dataset/HASCID-Dataset',
                                                                                            'dataset/HFD100 Mat dataset'], help='path to dataset')
    parser.add_argument('--train_mode', type=str, default='image', choices=['pixel', 'image'], 
                        help='training mode')
    parser.add_argument('--eval_ratio', type=float, default=0.1, help='the ratio of test data during training')
    parser.add_argument('--R-n', type=int, default=1, help='the number of random measurements, if None, then use full measurements')
    parser.add_argument('--dataset', type=str, choices=['HASCID', 'HFD'], default='HASCID', help='which dataset to use')
    
    # model parameters
    parser.add_argument('--spectral_channels', type=int, default=64, help='number of spectral bands of input hyperspectral data')
    parser.add_argument('--sensor_channels', type=int, default=30, help='number of channels of the sensor response')
    parser.add_argument('--base_channels', type=int, default=128, help='base channels of model')
    
    # visualization parameters
    parser.add_argument('--num_examples', type=int, default=5, help='number of examples to visualize')
    parser.add_argument('--model_path', type=str, default='results/2d_hsi_diffusion/HFD/R_1/l2_loss/checkpoint_epoch_101.pth', help='path to trained model checkpoint')
    parser.add_argument('--save_path', type=str, default='results/2d_visualization/HFD/R_1/l2_loss', help='path to save visualization results')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='dataset split to visualize')
    
    # generation parameters
    parser.add_argument('--num_steps', type=int, default=1000, help='number of denoising steps')
    parser.add_argument('--method', type=str, default='ddpm', choices=['ddpm', 'ddim'], help='sampling method')
    
    # visualization parameters
    parser.add_argument('--spatial_region', type=str, default='center', choices=['center', 'random', 'corner'],
                        help='which spatial region to analyze in detail')
    parser.add_argument('--num_spectral_bands', type=int, default=6, help='number of spectral bands to visualize')
    
    # device
    parser.add_argument('--device', type=str, default='cuda', help='device to use for computation')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    return parser


def load_model(model_path: str, device: torch.device, spectral_channels: int = 64, 
               sensor_channels: int = 30, base_channels: int = 128):
    """Load the trained 2D diffusion model"""
    model = U2NetHyperspectral(
        spectral_channels=spectral_channels,
        sensor_channels=sensor_channels,
        base_channels=base_channels,
    )
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    for k, v in model.named_parameters():
        if not torch.equal(v, checkpoint['model_state_dict'][k]):
            print(f"Warning: Parameter {k} does not match checkpoint value.")
            raise ValueError("Model parameters do not match checkpoint.")
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    return model


def generate_samples(model, diffusion_trainer, gt_data, sensor_data, device, num_steps=50, method='ddpm'):
    """Generate samples using the trained diffusion model"""
    model.eval()
    
    with torch.no_grad():
        # gt_data shape: [batch_size, spectral_channels, H, W]
        # sensor_data shape: [batch_size, sensor_channels, H, W]
        batch_size, spectral_channels, H, W = gt_data.shape
        
        # Generate samples
        generated = diffusion_trainer.sample(
            model=model,
            cond=sensor_data,
            shape=(batch_size, spectral_channels, H, W),
            n_steps=num_steps,
            method=method,
            progress=True
        )
        
    return generated


def extract_spatial_region(data, region_type='center', region_size=(32, 32)):
    """Extract a spatial region from hyperspectral data for detailed analysis"""
    B, C, H, W = data.shape
    h_size, w_size = region_size
    
    if region_type == 'center':
        h_start = max(0, (H - h_size) // 2)
        w_start = max(0, (W - w_size) // 2)
    elif region_type == 'corner':
        h_start, w_start = 0, 0
    elif region_type == 'random':
        h_start = random.randint(0, max(0, H - h_size))
        w_start = random.randint(0, max(0, W - w_size))
    else:
        h_start, w_start = 0, 0
    
    h_end = min(H, h_start + h_size)
    w_end = min(W, w_start + w_size)
    
    return data[:, :, h_start:h_end, w_start:w_end], (h_start, w_start, h_end, w_end)


def create_comparison_plot(gt_data, sensor_data, generated_data, wavelengths, sensor_wavelengths, 
                         example_idx, save_path, spatial_region='center', num_spectral_bands=6):
    """Create comprehensive comparison plots for a single example"""
    
    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    
    # Extract single image from batch
    gt_img = gt_data[0].cpu().numpy()  # [spectral_channels, H, W]
    sensor_img = sensor_data[0].cpu().numpy()  # [sensor_channels, H, W]
    generated_img = generated_data[0].cpu().numpy()  # [spectral_channels, H, W]
    
    C, H, W = gt_img.shape
    
    # Select RGB bands for visualization (assuming wavelengths cover visible range)
    if len(wavelengths) >= 3:
        # Find closest to RGB wavelengths (650nm Red, 550nm Green, 450nm Blue)
        rgb_indices = [
            np.argmin(np.abs(wavelengths - 650)),  # Red
            np.argmin(np.abs(wavelengths - 550)),  # Green
            np.argmin(np.abs(wavelengths - 450))   # Blue
        ]
    else:
        # Use first, middle, and last channels if wavelength info is not available
        rgb_indices = [0, C//2, C-1]
    
    # Create RGB images for visualization
    gt_rgb = np.stack([gt_img[rgb_indices[0]], gt_img[rgb_indices[1]], gt_img[rgb_indices[2]]], axis=-1)
    generated_rgb = np.stack([generated_img[rgb_indices[0]], generated_img[rgb_indices[1]], generated_img[rgb_indices[2]]], axis=-1)
    
    # Normalize RGB images to [0, 1]
    gt_rgb = np.clip(gt_rgb / np.max(gt_rgb), 0, 1)
    generated_rgb = np.clip(generated_rgb / np.max(generated_rgb), 0, 1)
    
    # Create the main comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Example {example_idx + 1}: 2D HSI Reconstruction Comparison', fontsize=16)
    
    # Plot 1: Ground Truth RGB
    axes[0, 0].imshow(gt_rgb)
    axes[0, 0].set_title('Ground Truth (RGB)')
    axes[0, 0].axis('off')
    
    # Plot 2: Generated RGB
    axes[0, 1].imshow(generated_rgb)
    axes[0, 1].set_title('Generated (RGB)')
    axes[0, 1].axis('off')
    
    # Plot 3: Difference RGB
    diff_rgb = np.abs(gt_rgb - generated_rgb)
    axes[0, 2].imshow(diff_rgb)
    axes[0, 2].set_title('Absolute Difference (RGB)')
    axes[0, 2].axis('off')
    
    # Plot 4: Selected spectral bands - Ground Truth
    band_indices = np.linspace(0, C-1, num_spectral_bands, dtype=int)
    for i, band_idx in enumerate(band_indices[:3]):  # Show first 3 bands
        ax = axes[1, i]
        im = ax.imshow(gt_img[band_idx], cmap='viridis')
        ax.set_title(f'GT Band {band_idx} ({wavelengths[band_idx]:.1f}nm)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save RGB comparison plot
    plot_path = os.path.join(save_path, f'comparison_2d_example_{example_idx + 1}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create spectral comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Example {example_idx + 1}: Spectral Band Comparison', fontsize=16)
    
    # Show more spectral bands comparison
    for i, band_idx in enumerate(band_indices[:6]):
        row = i // 3
        col = i % 3
        
        # Create subplot with GT and Generated side by side
        ax = axes[row, col]
        
        # Combine GT and Generated horizontally for comparison
        combined_img = np.concatenate([gt_img[band_idx], generated_img[band_idx]], axis=1)
        im = ax.imshow(combined_img, cmap='viridis')
        ax.set_title(f'Band {band_idx} ({wavelengths[band_idx]:.1f}nm)\nLeft: GT, Right: Generated')
        ax.axis('off')
        
        # Add a vertical line to separate GT and Generated
        ax.axvline(x=W-0.5, color='white', linewidth=2, linestyle='--')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save spectral comparison plot
    spectral_path = os.path.join(save_path, f'spectral_2d_example_{example_idx + 1}.png')
    plt.savefig(spectral_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pixel-wise spectral analysis
    create_spectral_analysis_plot(gt_img, generated_img, wavelengths, example_idx, save_path, spatial_region)
    
    # Calculate metrics
    mse = np.mean((gt_img - generated_img) ** 2)
    mae = np.mean(np.abs(gt_img - generated_img))
    psnr = 10 * np.log10(1 / mse) if mse > 0 else float('inf')
    
    # Spectral metrics (average across spatial dimensions)
    spectral_mse = np.mean(np.mean((gt_img - generated_img) ** 2, axis=(1, 2)))
    spectral_mae = np.mean(np.mean(np.abs(gt_img - generated_img), axis=(1, 2)))
    
    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr,
        'spectral_mse': spectral_mse,
        'spectral_mae': spectral_mae
    }


def create_spectral_analysis_plot(gt_img, generated_img, wavelengths, example_idx, save_path, spatial_region='center'):
    """Create detailed spectral analysis plots"""
    
    C, H, W = gt_img.shape
    
    # Extract a spatial region for detailed analysis
    region_size = (min(32, H), min(32, W))
    
    if spatial_region == 'center':
        h_start = max(0, (H - region_size[0]) // 2)
        w_start = max(0, (W - region_size[1]) // 2)
    elif spatial_region == 'corner':
        h_start, w_start = 0, 0
    else:  # random
        h_start = random.randint(0, max(0, H - region_size[0]))
        w_start = random.randint(0, max(0, W - region_size[1]))
    
    h_end = min(H, h_start + region_size[0])
    w_end = min(W, w_start + region_size[1])
    
    # Sample a few pixels for spectral profile comparison
    sample_pixels = [(h_start + i, w_start + j) for i in range(0, h_end-h_start, max(1, (h_end-h_start)//4))
                     for j in range(0, w_end-w_start, max(1, (w_end-w_start)//4))][:9]  # Max 9 pixels
    
    # Create spectral profile plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'Example {example_idx + 1}: Spectral Profile Analysis', fontsize=16)
    
    for idx, (h, w) in enumerate(sample_pixels):
        if idx >= 9:
            break
            
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        gt_spectrum = gt_img[:, h, w]
        generated_spectrum = generated_img[:, h, w]
        
        ax.plot(wavelengths, gt_spectrum, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(wavelengths, generated_spectrum, 'r--', linewidth=2, label='Generated', alpha=0.8)
        
        ax.set_title(f'Pixel ({h}, {w})')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Calculate and display pixel-wise metrics
        pixel_mse = np.mean((gt_spectrum - generated_spectrum) ** 2)
        pixel_mae = np.mean(np.abs(gt_spectrum - generated_spectrum))
        
        # Add metrics text
        metrics_text = f'MSE: {pixel_mse:.6f}\nMAE: {pixel_mae:.6f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(len(sample_pixels), 9):
        row = idx // 3
        col = idx % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save spectral analysis plot
    spectral_analysis_path = os.path.join(save_path, f'spectral_analysis_2d_example_{example_idx + 1}.png')
    plt.savefig(spectral_analysis_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_plot(all_metrics, save_path):
    """Create a summary plot showing metrics across all examples"""
    
    mse_values = [m['mse'] for m in all_metrics]
    mae_values = [m['mae'] for m in all_metrics]
    psnr_values = [m['psnr'] for m in all_metrics]
    spectral_mse_values = [m['spectral_mse'] for m in all_metrics]
    spectral_mae_values = [m['spectral_mae'] for m in all_metrics]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('2D HSI Diffusion Model - Performance Summary', fontsize=16)
    
    # MSE plot
    axes[0, 0].bar(range(1, len(mse_values) + 1), mse_values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Mean Squared Error (MSE)')
    axes[0, 0].set_xlabel('Example Number')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=np.mean(mse_values), color='red', linestyle='--', 
                      label=f'Average: {np.mean(mse_values):.6f}')
    axes[0, 0].legend()
    
    # MAE plot
    axes[0, 1].bar(range(1, len(mae_values) + 1), mae_values, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Mean Absolute Error (MAE)')
    axes[0, 1].set_xlabel('Example Number')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=np.mean(mae_values), color='red', linestyle='--', 
                      label=f'Average: {np.mean(mae_values):.6f}')
    axes[0, 1].legend()
    
    # PSNR plot
    axes[0, 2].bar(range(1, len(psnr_values) + 1), psnr_values, color='lightgreen', alpha=0.7)
    axes[0, 2].set_title('Peak Signal-to-Noise Ratio (PSNR)')
    axes[0, 2].set_xlabel('Example Number')
    axes[0, 2].set_ylabel('PSNR (dB)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=np.mean(psnr_values), color='red', linestyle='--', 
                      label=f'Average: {np.mean(psnr_values):.2f} dB')
    axes[0, 2].legend()
    
    # Spectral MSE plot
    axes[1, 0].bar(range(1, len(spectral_mse_values) + 1), spectral_mse_values, color='orange', alpha=0.7)
    axes[1, 0].set_title('Spectral MSE')
    axes[1, 0].set_xlabel('Example Number')
    axes[1, 0].set_ylabel('Spectral MSE')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=np.mean(spectral_mse_values), color='red', linestyle='--', 
                      label=f'Average: {np.mean(spectral_mse_values):.6f}')
    axes[1, 0].legend()
    
    # Spectral MAE plot
    axes[1, 1].bar(range(1, len(spectral_mae_values) + 1), spectral_mae_values, color='purple', alpha=0.7)
    axes[1, 1].set_title('Spectral MAE')
    axes[1, 1].set_xlabel('Example Number')
    axes[1, 1].set_ylabel('Spectral MAE')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=np.mean(spectral_mae_values), color='red', linestyle='--', 
                      label=f'Average: {np.mean(spectral_mae_values):.6f}')
    axes[1, 1].legend()
    
    # Combined metrics comparison
    axes[1, 2].plot(range(1, len(mse_values) + 1), mse_values, 'o-', label='MSE', alpha=0.7)
    axes[1, 2].plot(range(1, len(mae_values) + 1), mae_values, 's-', label='MAE', alpha=0.7)
    axes[1, 2].set_title('MSE vs MAE Comparison')
    axes[1, 2].set_xlabel('Example Number')
    axes[1, 2].set_ylabel('Error Value')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'summary_metrics_2d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\n2D HSI Diffusion Model Summary Statistics:")
    print(f"Average MSE: {np.mean(mse_values):.6f} ± {np.std(mse_values):.6f}")
    print(f"Average MAE: {np.mean(mae_values):.6f} ± {np.std(mae_values):.6f}")
    print(f"Average PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f} dB")
    print(f"Average Spectral MSE: {np.mean(spectral_mse_values):.6f} ± {np.std(spectral_mse_values):.6f}")
    print(f"Average Spectral MAE: {np.mean(spectral_mae_values):.6f} ± {np.std(spectral_mae_values):.6f}")
    print(f"Min MSE: {np.min(mse_values):.6f}")
    print(f"Max MSE: {np.max(mse_values):.6f}")
    print(f"Min PSNR: {np.min(psnr_values):.2f} dB")
    print(f"Max PSNR: {np.max(psnr_values):.2f} dB")


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
            train_mode='image',  # Force image mode for 2D
            split=args.split,
            eval_ratio=args.eval_ratio,
            data_format='image',
        )
    elif args.dataset == 'HFD':
        dataset = HFD_data(
            data_path=args.data_path,
            train_mode='image',  # Force image mode for 2D
            split=args.split,
            eval_ratio=args.eval_ratio,
            data_format='image',
            R_n=args.R_n,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Create dataloader with image collate function
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, 
                          collate_fn=image_collate_fn)
    
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # Load model
    model = load_model(args.model_path, device, args.spectral_channels, 
                      args.sensor_channels, args.base_channels)
    
    # Initialize diffusion trainer
    diffusion_trainer = DiffusionTrainer(device=device)
    
    # Get wavelengths from dataset
    wavelengths = dataset.wavelens[:args.spectral_channels] if args.dataset == 'HASCID' else dataset.new_wavelens[:args.spectral_channels]
    
    # For sensor data, create appropriate wavelength array
    sensor_wavelengths = np.linspace(400, 1000, args.sensor_channels)
    
    print(f"Generating visualizations for {args.num_examples} examples...")
    
    all_metrics = []
    examples_processed = 0
    
    # Generate and visualize examples
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing examples")):
        if examples_processed >= args.num_examples:
            break
            
        # Extract data from batch
        gt_data, sensor_data = batch
        
        # Move to device
        gt_data = gt_data.to(device)
        sensor_data = sensor_data.to(device)
        
        print(f"\nProcessing example {examples_processed + 1}")
        print(f"GT data shape: {gt_data.shape}")
        print(f"Sensor data shape: {sensor_data.shape}")
        print(f"GT data range: [{gt_data.min():.3f}, {gt_data.max():.3f}]")
        print(f"Sensor data range: [{sensor_data.min():.3f}, {sensor_data.max():.3f}]")
        
        # Generate samples
        with torch.no_grad():
            generated_data = generate_samples(
                model, diffusion_trainer, gt_data, sensor_data, device, 
                num_steps=args.num_steps, method=args.method
            )
        
        print(f"Generated data shape: {generated_data.shape}")
        print(f"Generated data range: [{generated_data.min():.3f}, {generated_data.max():.3f}]")
        
        # Create comparison plots
        example_metrics = create_comparison_plot(
            gt_data, sensor_data, generated_data,
            wavelengths, sensor_wavelengths,
            examples_processed, args.save_path, 
            args.spatial_region, args.num_spectral_bands
        )
        
        all_metrics.append(example_metrics)
        examples_processed += 1
        
        print(f"Example {examples_processed} metrics:")
        print(f"  MSE: {example_metrics['mse']:.6f}")
        print(f"  MAE: {example_metrics['mae']:.6f}")
        print(f"  PSNR: {example_metrics['psnr']:.2f} dB")
        print(f"  Spectral MSE: {example_metrics['spectral_mse']:.6f}")
        print(f"  Spectral MAE: {example_metrics['spectral_mae']:.6f}")
    
    # Create summary plot
    create_summary_plot(all_metrics, args.save_path)
    
    # Save metrics to file
    metrics_file = os.path.join(args.save_path, 'metrics_2d.txt')
    with open(metrics_file, 'w') as f:
        f.write("Example\tMSE\tMAE\tPSNR\tSpectral_MSE\tSpectral_MAE\n")
        for i, metrics in enumerate(all_metrics):
            f.write(f"{i+1}\t{metrics['mse']:.6f}\t{metrics['mae']:.6f}\t{metrics['psnr']:.2f}\t"
                   f"{metrics['spectral_mse']:.6f}\t{metrics['spectral_mae']:.6f}\n")
        
        # Add averages
        f.write(f"\nAverage\t{np.mean([m['mse'] for m in all_metrics]):.6f}\t"
               f"{np.mean([m['mae'] for m in all_metrics]):.6f}\t"
               f"{np.mean([m['psnr'] for m in all_metrics]):.2f}\t"
               f"{np.mean([m['spectral_mse'] for m in all_metrics]):.6f}\t"
               f"{np.mean([m['spectral_mae'] for m in all_metrics]):.6f}\n")
    
    print(f"\n2D Visualization complete! Results saved to: {args.save_path}")
    print(f"- Individual comparison plots: comparison_2d_example_*.png")
    print(f"- Spectral band plots: spectral_2d_example_*.png")
    print(f"- Spectral analysis plots: spectral_analysis_2d_example_*.png")
    print(f"- Summary metrics plot: summary_metrics_2d.png")
    print(f"- Detailed metrics: metrics_2d.txt")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    print("2D HSI Diffusion Visualization Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    visualize_results(args)