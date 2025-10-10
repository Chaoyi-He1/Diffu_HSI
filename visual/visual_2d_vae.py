import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from torch.utils.data import DataLoader
import random
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.my_dataset import HASCID_data, image_collate_fn
from model.hyperspectral_vae import HyperspectralVAE, vae_loss
from model.latent_u2net_hyperspectral import LatentU2NetHyperspectral
from model.diffusion_trainer import DiffusionTrainer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Visualization for 2D HSI Diffusion Model with VAE')
    
    # dataset parameters
    parser.add_argument('--data_path', type=str, default='dataset/HASCID-Dataset', help='path to dataset')
    parser.add_argument('--train_mode', type=str, default='image', choices=['pixel', 'image'], 
                        help='training mode (use image for VAE)')
    parser.add_argument('--eval_ratio', type=float, default=0.1, help='the ratio of test data during training')
    parser.add_argument('--R-n', type=int, default=None, help='the number of random measurements, if None, then use full measurements')
    parser.add_argument('--dataset', type=str, default='HASCID', choices=['HASCID', 'HFD'], help='dataset to use')
    
    # model parameters
    # VAE parameters
    parser.add_argument('--vae_latent_channels', type=int, default=12, help='dimension of latent space of VAE compression')
    parser.add_argument('--vae_base_channels', type=int, default=128, help='base channels of VAE model')
    parser.add_argument('--input_channels', type=int, default=160, help='number of spectral bands of input data')
    parser.add_argument('--vae_model_path', type=str, default='results/vae_hyperspectral/base_128_latent_12/vae_final_model.pth',
                        help='path to trained VAE model')
    
    # Diffusion model parameters
    parser.add_argument('--sensor_channels', type=int, default=30, help='number of channels of the sensor response (condition)')
    parser.add_argument('--diffusion_base_channels', type=int, default=128, help='base channels of diffusion U-Net model')
    
    # visualization parameters
    parser.add_argument('--num_examples', type=int, default=5, help='number of examples to visualize')
    parser.add_argument('--model_path', type=str, default='results/2d_hsi_diffusion_vae/best_model.pth', 
                        help='path to trained diffusion model checkpoint')
    parser.add_argument('--save_path', type=str, default='results/2d_vae_visualization', 
                        help='path to save visualization results')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], 
                        help='dataset split to visualize')
    
    # generation parameters
    parser.add_argument('--num_steps', type=int, default=50, help='number of denoising steps')
    parser.add_argument('--method', type=str, default='ddim', choices=['ddpm', 'ddim'], help='sampling method')
    
    # visualization parameters
    parser.add_argument('--spatial_region', type=str, default='center', choices=['center', 'random', 'corner'],
                        help='which spatial region to analyze in detail')
    parser.add_argument('--spectral_bands', type=str, default='selected', choices=['all', 'selected', 'rgb'],
                        help='which spectral bands to visualize')
    parser.add_argument('--kl_weight', type=float, default=1e-6, help='KL weight used during training')
    parser.add_argument('--num_spectral_bands', type=int, default=6, help='number of spectral bands to visualize')
    
    # device
    parser.add_argument('--device', type=str, default='cuda', help='device to use for computation')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    return parser


def load_vae_model(model_path: str, device: torch.device, spectral_channels: int = 160, 
                   latent_channels: int = 12, base_channels: int = 128):
    """Load the trained VAE model"""
    model = HyperspectralVAE(
        spectral_channels=spectral_channels,
        latent_channels=latent_channels,
        base_channels=base_channels,
    )
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"VAE model checkpoint not found at {model_path}")
    
    print(f"Loading VAE model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Warning: Failed to load with default settings: {e}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded VAE checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("VAE model loaded successfully")
    
    return model


def load_diffusion_model(model_path: str, device: torch.device, sensor_channels: int = 30,
                        latent_channels: int = 12, base_channels: int = 128):
    """Load the trained diffusion model"""
    model = LatentU2NetHyperspectral(
        sensor_channels=sensor_channels,
        latent_channels=latent_channels,
        base_channels=base_channels
    )
    vae = HyperspectralVAE(
        spectral_channels=160,
        latent_channels=latent_channels,
        base_channels=base_channels,
    )
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Diffusion model checkpoint not found at {model_path}")
    
    print(f"Loading diffusion model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Warning: Failed to load with default settings: {e}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    if 'vae_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['vae_state_dict'])
    else:
        print("Warning: VAE state dict not found in diffusion checkpoint.")
        raise ValueError("VAE state dict not found in diffusion checkpoint.")
    
    for k, v in model.named_parameters():
        if not torch.equal(v, checkpoint['model_state_dict'][k]):
            print(f"Parameter {k} not loaded correctly.")
            raise ValueError(f"Parameter {k} not loaded correctly.")
    print("All diffusion model parameters loaded successfully.")
    for k, v in vae.named_parameters():
        if not torch.equal(v, checkpoint['vae_state_dict'][k]):
            print(f"Parameter {k} in VAE not loaded correctly.")
            raise ValueError(f"Parameter {k} in VAE not loaded correctly.")
    print("All VAE parameters loaded successfully.")    
    
    model = model.to(device)
    model.eval()
    vae = vae.to(device)
    vae.eval()
    
    return model, vae


def generate_with_latent_diffusion(diffusion_model, vae_model, sensor_data, device, 
                                   num_steps=1000, method='ddim'):
    """
    Generate hyperspectral images using latent diffusion model
    
    Args:
        diffusion_model: Trained latent diffusion model
        vae_model: Trained VAE model for encoding/decoding
        sensor_data: Sensor response data [B, sensor_channels, H, W]
        device: Device to run on
        num_steps: Number of denoising steps
        method: Sampling method ('ddim' or 'ddpm')
    
    Returns:
        generated_data: Generated hyperspectral images [B, spectral_channels, H, W]
        generated_latents: Generated latent representations [B, latent_channels, H/8, W/8]
    """
    diffusion_model.eval()
    vae_model.eval()
    
    with torch.no_grad():
        # Get latent dimensions from sensor data
        B, _, H, W = sensor_data.shape
        latent_h, latent_w = H // 8, W // 8  # VAE downsamples by 8x
        latent_channels = vae_model.latent_channels
        
        # Create diffusion trainer for sampling
        diffusion_trainer = DiffusionTrainer(device=device)
        
        # Generate samples using the trainer's unified sampling method
        generated_latents = diffusion_trainer.sample(
            model=diffusion_model,
            cond=sensor_data,
            shape=(B, latent_channels, latent_h, latent_w),
            n_steps=num_steps,
            method=method,
            progress=True
        )
        
        # Decode latents back to hyperspectral space
        # Scale latents back (reverse the scaling done during training)
        generated_latents_scaled = generated_latents / 0.024
        generated_data = vae_model.decode(generated_latents_scaled)
        
    return generated_data, generated_latents


def create_diffusion_comparison_plot(gt_data, generated_data, sensor_data, wavelengths, 
                                   example_idx, save_path, spatial_region='center'):
    """
    Create comprehensive comparison plots for diffusion generation vs ground truth
    """
    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    B, C, H, W = gt_data.shape
    gt_data = gt_data[0]  # [C, H, W]
    generated_data = generated_data[0]  # [C, H, W]
    sensor_data = sensor_data[0]  # [sensor_channels, H, W]
    
    # Select RGB bands (assuming wavelengths in nm)
    rgb_indices = [np.argmin(np.abs(wavelengths - 650)),  # Red
                   np.argmin(np.abs(wavelengths - 550)),  # Green
                   np.argmin(np.abs(wavelengths - 450))]  # Blue
    
    rgb_gt = gt_data[rgb_indices, :, :].cpu().numpy()  # [3, H, W]
    rgb_gen = generated_data[rgb_indices, :, :].cpu().numpy()  # [3, H, W]
    
    # Normalize RGB images
    rgb_gt = np.clip(rgb_gt / np.max(rgb_gt), 0, 1).transpose(1, 2, 0)  # [H, W, 3]
    rgb_gen = np.clip(rgb_gen / np.max(rgb_gen), 0, 1).transpose(1, 2, 0)  # [H, W, 3]
    
    # Create RGB comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # First row: RGB comparison
    axes[0, 0].imshow(rgb_gt)
    axes[0, 0].set_title('Ground Truth RGB')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rgb_gen)
    axes[0, 1].set_title('Generated RGB')
    axes[0, 1].axis('off')
    
    # Difference image
    diff_image = np.abs(rgb_gt - rgb_gen)
    axes[0, 2].imshow(diff_image / np.max(diff_image))
    axes[0, 2].set_title('Absolute Difference RGB')
    axes[0, 2].axis('off')
    
    # Error maps
    mse_map = torch.mean((gt_data - generated_data) ** 2, dim=0).cpu().numpy()
    mae_map = torch.mean(torch.abs(gt_data - generated_data), dim=0).cpu().numpy()
    
    im1 = axes[1, 1].imshow(mse_map, cmap='hot')
    axes[1, 1].set_title('MSE Error Map')
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1])
    
    im2 = axes[1, 2].imshow(mae_map, cmap='hot')
    axes[1, 2].set_title('MAE Error Map')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2])
    
    plt.suptitle(f'Example {example_idx + 1} - Diffusion Generation Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'diffusion_comparison_example_{example_idx + 1}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create spectral profile comparison plot
    selected_pixels = random.sample([(i, j) for i in range(H) for j in range(W)], 2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    for idx, pixel in enumerate(selected_pixels):
        i, j = pixel
        
        # Plot GT vs Generated spectra
        axes[idx, 0].plot(wavelengths, gt_data[:, i, j].cpu().numpy(), 
                         label=f'GT Pixel ({i},{j})', linewidth=2)
        axes[idx, 0].plot(wavelengths, generated_data[:, i, j].cpu().numpy(), 
                         label=f'Generated Pixel ({i},{j})', linestyle='--', linewidth=2)
        axes[idx, 0].set_title(f'Spectral Profile at Pixel ({i},{j})')
        axes[idx, 0].set_xlabel('Wavelength (nm)')
        axes[idx, 0].set_ylabel('Reflectance')
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].legend()
        
        # Plot absolute difference
        axes[idx, 1].plot(wavelengths, 
                         np.abs(gt_data[:, i, j].cpu().numpy() - generated_data[:, i, j].cpu().numpy()),
                         label=f'Abs Diff Pixel ({i},{j})', color='red', linewidth=2)
        axes[idx, 1].set_title(f'Spectral Absolute Difference at Pixel ({i},{j})')
        axes[idx, 1].set_xlabel('Wavelength (nm)')
        axes[idx, 1].set_ylabel('Absolute Difference')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].legend()
    
    plt.suptitle(f'Example {example_idx + 1} - Spectral Profile Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'spectral_analysis_example_{example_idx + 1}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute and return metrics
    mse = torch.mean((gt_data - generated_data) ** 2).item()
    mae = torch.mean(torch.abs(gt_data - generated_data)).item()
    psnr = 10 * np.log10(1 / mse) if mse > 0 else float('inf')
    
    # Spectral metrics
    spectral_mse = torch.mean(torch.mean((gt_data - generated_data) ** 2, dim=(1,2))).cpu().item()
    spectral_mae = torch.mean(torch.mean(torch.abs(gt_data - generated_data), dim=(1,2))).cpu().item()
    
    # Compute spectral correlation
    gt_flat = gt_data.view(gt_data.size(0), -1)  # [C, H*W]
    gen_flat = generated_data.view(generated_data.size(0), -1)  # [C, H*W]
    spectral_corr = torch.corrcoef(torch.stack([gt_flat.mean(dim=1), gen_flat.mean(dim=1)]))[0, 1].item()
    
    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr,
        'spectral_mse': spectral_mse,
        'spectral_mae': spectral_mae,
        'spectral_correlation': spectral_corr
    }


def create_latent_generation_analysis(generated_latents, vae_model, example_idx, save_path):
    """Create plots to analyze the generated latents"""
    latent_np = generated_latents[0].cpu().numpy()  # [latent_channels, H/8, W/8]
    C, H, W = latent_np.shape
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Visualize first 4 latent channels
    for i in range(min(4, C)):
        axes[0, i].imshow(latent_np[i], cmap='RdBu', aspect='auto')
        axes[0, i].set_title(f'Generated Latent Ch.{i+1}')
        axes[0, i].axis('off')
    
    # Visualize latent statistics
    latent_means = np.mean(latent_np, axis=(1, 2))
    latent_stds = np.std(latent_np, axis=(1, 2))
    
    axes[1, 0].bar(range(len(latent_means)), latent_means)
    axes[1, 0].set_title('Latent Channel Means')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Mean Value')
    
    axes[1, 1].bar(range(len(latent_stds)), latent_stds)
    axes[1, 1].set_title('Latent Channel Stds')
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('Std Value')
    
    # Latent channel correlation matrix
    latent_flat = latent_np.reshape(C, -1)  # [C, H*W]
    corr_matrix = np.corrcoef(latent_flat)
    im = axes[1, 2].imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 2].set_title('Latent Channel Correlation')
    plt.colorbar(im, ax=axes[1, 2])
    
    # Latent distribution histogram
    axes[1, 3].hist(latent_np.flatten(), bins=50, alpha=0.7, density=True)
    axes[1, 3].set_title('Latent Value Distribution')
    axes[1, 3].set_xlabel('Latent Value')
    axes[1, 3].set_ylabel('Density')
    
    plt.suptitle(f'Generated Latent Analysis - Example {example_idx + 1}', fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(save_path, f'latent_generation_analysis_example_{example_idx + 1}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'latent_mean_norm': np.mean(np.linalg.norm(latent_flat, axis=1)),
        'latent_std_mean': np.mean(latent_stds),
        'latent_range': (float(latent_np.min()), float(latent_np.max()))
    }


def create_summary_plot(all_metrics, save_path):
    """Create a summary plot showing metrics across all examples"""
    
    # Extract metrics
    mse_values = [m['mse'] for m in all_metrics]
    mae_values = [m['mae'] for m in all_metrics]
    psnr_values = [m['psnr'] for m in all_metrics]
    spectral_corr_values = [m['spectral_correlation'] for m in all_metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # MSE plot
    axes[0, 0].bar(range(1, len(mse_values) + 1), mse_values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Mean Squared Error (MSE)')
    axes[0, 0].set_xlabel('Example Number')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=np.mean(mse_values), color='red', linestyle='--', 
                      label=f'Avg: {np.mean(mse_values):.6f}')
    axes[0, 0].legend()
    
    # MAE plot
    axes[0, 1].bar(range(1, len(mae_values) + 1), mae_values, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Mean Absolute Error (MAE)')
    axes[0, 1].set_xlabel('Example Number')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=np.mean(mae_values), color='red', linestyle='--', 
                      label=f'Avg: {np.mean(mae_values):.6f}')
    axes[0, 1].legend()
    
    # PSNR plot
    axes[1, 0].bar(range(1, len(psnr_values) + 1), psnr_values, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Peak Signal-to-Noise Ratio (PSNR)')
    axes[1, 0].set_xlabel('Example Number')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=np.mean(psnr_values), color='red', linestyle='--', 
                      label=f'Avg: {np.mean(psnr_values):.2f} dB')
    axes[1, 0].legend()
    
    # Spectral Correlation plot
    axes[1, 1].bar(range(1, len(spectral_corr_values) + 1), spectral_corr_values, color='purple', alpha=0.7)
    axes[1, 1].set_title('Spectral Correlation')
    axes[1, 1].set_xlabel('Example Number')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=np.mean(spectral_corr_values), color='red', linestyle='--', 
                      label=f'Avg: {np.mean(spectral_corr_values):.4f}')
    axes[1, 1].legend()
    
    plt.suptitle('Diffusion Generation Metrics Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'diffusion_summary_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\nDiffusion Generation Summary Statistics:")
    print(f"Average MSE: {np.mean(mse_values):.6f} ± {np.std(mse_values):.6f}")
    print(f"Average MAE: {np.mean(mae_values):.6f} ± {np.std(mae_values):.6f}")
    print(f"Average PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f} dB")
    print(f"Average Spectral Correlation: {np.mean(spectral_corr_values):.4f} ± {np.std(spectral_corr_values):.4f}")


def visualize_latent_diffusion_results(args):
    """Main visualization function for latent diffusion model"""
    
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
    dataset = HASCID_data(
        data_path=args.data_path,
        train_mode='image',
        split=args.split,
        eval_ratio=args.eval_ratio,
        data_format='image'
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, 
                          collate_fn=image_collate_fn)
    
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # Load models
    print("Loading models...")
    vae_model = load_vae_model(args.vae_model_path, device, 
                              args.input_channels, args.vae_latent_channels, args.vae_base_channels)
    
    diffusion_model = load_diffusion_model(args.model_path, device,
                                         args.sensor_channels, args.vae_latent_channels, args.diffusion_base_channels)
    
    # Get wavelengths from dataset
    wavelengths = dataset.wavelens[:args.input_channels]
    
    print(f"Generating visualizations for {args.num_examples} examples...")
    
    all_metrics = []
    all_latent_stats = []
    examples_processed = 0
    
    # Generate and visualize examples
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing examples")):
        if examples_processed >= args.num_examples:
            break
        
        # Extract data from batch
        gt_data, sensor_data = batch  # gt_data: [B, C, H, W], sensor_data: [B, N, H, W]
        
        # Move to device
        gt_data = gt_data.to(device)
        sensor_data = sensor_data.to(device)
        
        print(f"\nProcessing example {examples_processed + 1}")
        print(f"GT data shape: {gt_data.shape}")
        print(f"Sensor data shape: {sensor_data.shape}")
        print(f"GT data range: [{gt_data.min():.3f}, {gt_data.max():.3f}]")
        print(f"Sensor data range: [{sensor_data.min():.3f}, {sensor_data.max():.3f}]")
        
        # Generate with latent diffusion
        generated_data, generated_latents = generate_with_latent_diffusion(
            diffusion_model, vae_model, sensor_data, device, 
            args.num_steps, args.method
        )
        
        print(f"Generated data shape: {generated_data.shape}")
        print(f"Generated data range: [{generated_data.min():.3f}, {generated_data.max():.3f}]")
        print(f"Generated latents shape: {generated_latents.shape}")
        
        # Create comparison plot
        example_metrics = create_diffusion_comparison_plot(
            gt_data, generated_data, sensor_data, wavelengths, 
            examples_processed, args.save_path, args.spatial_region
        )
        all_metrics.append(example_metrics)
        
        # Create latent space analysis
        latent_stats = create_latent_generation_analysis(
            generated_latents, vae_model, examples_processed, args.save_path
        )
        all_latent_stats.append(latent_stats)
        
        examples_processed += 1
        
        print(f"Example {examples_processed} metrics:")
        print(f"  MSE: {example_metrics['mse']:.6f}")
        print(f"  MAE: {example_metrics['mae']:.6f}")
        print(f"  PSNR: {example_metrics['psnr']:.2f} dB")
        print(f"  Spectral Correlation: {example_metrics['spectral_correlation']:.4f}")
    
    # Create summary plot
    create_summary_plot(all_metrics, args.save_path)
    
    # Save detailed metrics to file
    metrics_file = os.path.join(args.save_path, 'diffusion_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Example\tMSE\tMAE\tPSNR\tSpectral_MSE\tSpectral_MAE\tSpectral_Correlation\n")
        for i, metrics in enumerate(all_metrics):
            f.write(f"{i+1}\t{metrics['mse']:.6f}\t{metrics['mae']:.6f}\t{metrics['psnr']:.2f}\t"
                   f"{metrics['spectral_mse']:.6f}\t{metrics['spectral_mae']:.6f}\t"
                   f"{metrics['spectral_correlation']:.4f}\n")
        
        # Add averages
        f.write(f"\nAverage\t{np.mean([m['mse'] for m in all_metrics]):.6f}\t"
               f"{np.mean([m['mae'] for m in all_metrics]):.6f}\t"
               f"{np.mean([m['psnr'] for m in all_metrics]):.2f}\t"
               f"{np.mean([m['spectral_mse'] for m in all_metrics]):.6f}\t"
               f"{np.mean([m['spectral_mae'] for m in all_metrics]):.6f}\t"
               f"{np.mean([m['spectral_correlation'] for m in all_metrics]):.4f}\n")
    
    # Save latent space statistics
    latent_file = os.path.join(args.save_path, 'latent_generation_stats.txt')
    with open(latent_file, 'w') as f:
        f.write("Example\tLatent_Mean_Norm\tLatent_Std_Mean\tLatent_Min\tLatent_Max\n")
        for i, stats in enumerate(all_latent_stats):
            f.write(f"{i+1}\t{stats['latent_mean_norm']:.6f}\t{stats['latent_std_mean']:.6f}\t"
                   f"{stats['latent_range'][0]:.6f}\t{stats['latent_range'][1]:.6f}\n")
    
    print(f"\nLatent diffusion visualization complete! Results saved to: {args.save_path}")
    print(f"- Individual comparison plots: diffusion_comparison_example_*.png")
    print(f"- Spectral analysis plots: spectral_analysis_example_*.png")
    print(f"- Latent analysis plots: latent_generation_analysis_example_*.png")
    print(f"- Summary metrics plot: diffusion_summary_metrics.png")
    print(f"- Detailed metrics: diffusion_metrics.txt")
    print(f"- Latent statistics: latent_generation_stats.txt")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    print("Latent Diffusion Visualization Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    visualize_latent_diffusion_results(args)


