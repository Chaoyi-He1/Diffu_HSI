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


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Visualization for Hyperspectral VAE Model')
    
    # dataset parameters
    parser.add_argument('--data_path', type=str, default='dataset/HASCID-Dataset', help='path to dataset')
    parser.add_argument('--train_mode', type=str, default='image', choices=['pixel', 'image'], 
                        help='training mode (use image for VAE)')
    parser.add_argument('--eval_ratio', type=float, default=0.1, help='the ratio of test data during training')
    
    # model parameters
    parser.add_argument('--spectral_channels', type=int, default=160, help='number of input spectral channels')
    parser.add_argument('--latent_channels', type=int, default=12, help='number of latent channels')
    parser.add_argument('--base_channels', type=int, default=128, help='base channels of model')
    
    # visualization parameters
    parser.add_argument('--num_examples', type=int, default=5, help='number of examples to visualize')
    parser.add_argument('--model_path', type=str, default='results/vae_hyperspectral/base_128_latent_12/vae_final_model.pth', 
                        help='path to trained model checkpoint')
    parser.add_argument('--save_path', type=str, default='results/vae_visualization', 
                        help='path to save visualization results')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], 
                        help='dataset split to visualize')
    
    # visualization parameters
    parser.add_argument('--spatial_region', type=str, default='center', choices=['center', 'random', 'corner'],
                        help='which spatial region to analyze in detail')
    parser.add_argument('--spectral_bands', type=str, default='selected', choices=['all', 'selected', 'rgb'],
                        help='which spectral bands to visualize')
    parser.add_argument('--kl_weight', type=float, default=1e-6, help='KL weight used during training')
    
    # device
    parser.add_argument('--device', type=str, default='cuda', help='device to use for computation')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    return parser


def load_vae_model(model_path: str, device: torch.device, spectral_channels: int = 160, 
                   latent_channels: int = 8, base_channels: int = 128):
    """Load the trained VAE model"""
    model = HyperspectralVAE(
        spectral_channels=spectral_channels,
        latent_channels=latent_channels,
        base_channels=base_channels,
    )
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    print(f"Loading VAE model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Warning: Failed to load with default settings: {e}")
        # Try with weights_only=False for compatibility
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Check if model config matches checkpoint
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            print(f"Checkpoint model config: {config}")
            
            # Warn if there are mismatches
            if config.get('spectral_channels', spectral_channels) != spectral_channels:
                print(f"Warning: spectral_channels mismatch. Using checkpoint value: {config['spectral_channels']}")
            if config.get('latent_channels', latent_channels) != latent_channels:
                print(f"Warning: latent_channels mismatch. Using checkpoint value: {config['latent_channels']}")
            if config.get('base_channels', base_channels) != base_channels:
                print(f"Warning: base_channels mismatch. Using checkpoint value: {config['base_channels']}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("VAE model loaded successfully")
    
    return model


def reconstruct_with_vae(model, input_data, device, sample=True):
    """Reconstruct hyperspectral images using the trained VAE model"""
    model.eval()
    
    with torch.no_grad():
        # input_data shape: [batch_size, spectral_channels, H, W] 
        input_data = input_data.to(device)
        
        # VAE forward pass
        reconstructed, mean, logvar = model(input_data, sample=sample)
        
        # Compute VAE loss components
        total_loss, recon_loss, kl_loss = vae_loss(reconstructed, input_data, mean, logvar)
        
    return reconstructed, mean, logvar, {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item()
    }


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


def create_vae_comparison_plot(gt_data, reconstructed_data, wavelengths, example_idx, save_path,
                               metrics_dict, spatial_region='center'):
    """
    Create comprehensive comparison plots for VAE reconstruction
    Pick the RGB bands from the wavelengths, then plot GT vs Reconstructed as RGB images
    Then randomly select 2 pixels and plot their full spectra of GT vs Reconstructed
    """
    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    B, C, H, W = gt_data.shape
    gt_data = gt_data[0]    # [C, H, W]
    reconstructed_data = reconstructed_data[0]  # [C, H, W]
    
    # Select RGB bands (assuming wavelengths in nm)
    rgb_indices = [np.argmin(np.abs(wavelengths - 650)),  # Red
                   np.argmin(np.abs(wavelengths - 550)),  # Green
                   np.argmin(np.abs(wavelengths - 450))]  # Blue
    rgb_gt = gt_data[rgb_indices, :, :].cpu().numpy()   # [3, H, W]
    rgb_recon = reconstructed_data[rgb_indices, :, :].cpu().numpy() # [3, H, W]
    rgb_gt = np.clip(rgb_gt / np.max(rgb_gt), 0, 1).transpose(1, 2, 0)  # [H, W, 3]
    rgb_recon = np.clip(rgb_recon / np.max(rgb_recon), 0, 1).transpose(1, 2, 0)  # [H, W, 3]
    
    # plot RGB for whole image
    fig, axes = plt.subplots(1, 3, figsize=(18, 12))
    axes[0].imshow(rgb_gt)
    axes[0].set_title('Ground Truth RGB')
    axes[0].axis('off')

    axes[1].imshow(rgb_recon)
    axes[1].set_title('VAE Reconstructed RGB')
    axes[1].axis('off')

    # Difference image
    diff_image = np.abs(rgb_gt - rgb_recon)
    axes[2].imshow(diff_image / np.max(diff_image))
    axes[2].set_title('Absolute Difference RGB')
    axes[2].axis('off')
    
    # Save the RGB comparison plot
    plt.suptitle(f'Example {example_idx + 1} - RGB Comparison', fontsize=16)
    plt.savefig(os.path.join(save_path, f'rgb_comparison_example_{example_idx + 1}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Analyze spectral profiles at randomly selected 2 pixels
    selected_pixels = random.sample([(i, j) for i in range(H) for j in range(W)], 2)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for idx, pixel in enumerate(selected_pixels):
        i, j = pixel
        axes[idx, 0].plot(wavelengths, gt_data[:, i, j].cpu().numpy(), label=f'GT Pixel ({i},{j})')
        axes[idx, 0].set_title('Spectral Profiles at Selected Pixels')
        axes[idx, 0].set_xlabel('Wavelength (nm)')
        axes[idx, 0].set_ylabel('Reflectance')
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].legend()

        axes[idx, 1].plot(wavelengths, reconstructed_data[:, i, j].cpu().numpy(), label=f'Recon Pixel ({i},{j})', linestyle='--')
        axes[idx, 1].set_title('Spectral Profiles at Selected Pixels')
        axes[idx, 1].set_xlabel('Wavelength (nm)')
        axes[idx, 1].set_ylabel('Reflectance')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].legend()

        axes[idx, 2].plot(wavelengths, np.abs(gt_data[:, i, j].cpu().numpy() - reconstructed_data[:, i, j].cpu().numpy()),
                          label=f'Abs Diff Pixel ({i},{j})', color='red')
        axes[idx, 2].set_title('Absolute Difference at Selected Pixels')
        axes[idx, 2].set_xlabel('Wavelength (nm)')
        axes[idx, 2].set_ylabel('Absolute Difference')
        axes[idx, 2].grid(True, alpha=0.3)
        axes[idx, 2].legend()
    
    plt.suptitle(f'Example {example_idx + 1} - Spectral Profiles', fontsize=16)
    plt.savefig(os.path.join(save_path, f'spectral_profiles_example_{example_idx + 1}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute metrics
    mse = torch.mean((gt_data - reconstructed_data) ** 2).item()
    mae = torch.mean(torch.abs(gt_data - reconstructed_data)).item()
    psnr = 10 * np.log10(1 / mse) if mse > 0 else float('inf')
    recon_loss = metrics_dict.get('recon_loss', 0)
    kl_loss = metrics_dict.get('kl_loss', 0)
    total_loss = metrics_dict.get('total_loss', 0)
    # Compute spectral metrics (average across spatial dimensions, then across spectral channels)
    spectral_mse = torch.mean(torch.mean((gt_data - reconstructed_data) ** 2, dim=(1,2))).cpu().item()  # scalar
    spectral_mae = torch.mean(torch.mean(torch.abs(gt_data - reconstructed_data), dim=(1,2))).cpu().item()  # scalar
    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'total_loss': total_loss,
        'spectral_mse': spectral_mse,
        'spectral_mae': spectral_mae
    }


def create_latent_analysis_plot(mean, logvar, example_idx, save_path):
    """Create plots to analyze the latent space"""
    mean_np = mean[0].cpu().numpy()  # [latent_channels, H/8, W/8]
    logvar_np = logvar[0].cpu().numpy()
    std_np = np.exp(0.5 * logvar_np)
    
    C, H, W = mean_np.shape
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Visualize first 4 latent channels - means
    for i in range(min(4, C)):
        axes[0, i].imshow(mean_np[i], cmap='RdBu', aspect='auto')
        axes[0, i].set_title(f'Latent Mean Ch.{i+1}')
        axes[0, i].axis('off')
    
    # Visualize first 4 latent channels - standard deviations
    for i in range(min(4, C)):
        axes[1, i].imshow(std_np[i], cmap='viridis', aspect='auto')
        axes[1, i].set_title(f'Latent Std Ch.{i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Latent Space Analysis - Example {example_idx + 1}', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_path, f'latent_analysis_example_{example_idx + 1}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return latent statistics
    return {
        'latent_mean_norm': np.mean(np.linalg.norm(mean_np.reshape(C, -1), axis=1)),
        'latent_std_mean': np.mean(std_np),
        'latent_compression_ratio': (mean_np.size) / (mean_np.shape[1] * mean_np.shape[2] * 160)  # Assuming 160 input channels
    }


def create_summary_plot(all_metrics, save_path):
    """Create a summary plot showing metrics across all examples"""
    
    # Extract metrics
    mse_values = [m['mse'] for m in all_metrics]
    mae_values = [m['mae'] for m in all_metrics]
    psnr_values = [m['psnr'] for m in all_metrics]
    recon_loss_values = [m['recon_loss'] for m in all_metrics]
    kl_loss_values = [m['kl_loss'] for m in all_metrics]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
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
    axes[0, 2].bar(range(1, len(psnr_values) + 1), psnr_values, color='lightgreen', alpha=0.7)
    axes[0, 2].set_title('Peak Signal-to-Noise Ratio (PSNR)')
    axes[0, 2].set_xlabel('Example Number')
    axes[0, 2].set_ylabel('PSNR (dB)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=np.mean(psnr_values), color='red', linestyle='--', 
                      label=f'Avg: {np.mean(psnr_values):.2f} dB')
    axes[0, 2].legend()
    
    # Reconstruction Loss plot
    axes[1, 0].bar(range(1, len(recon_loss_values) + 1), recon_loss_values, color='orange', alpha=0.7)
    axes[1, 0].set_title('Reconstruction Loss')
    axes[1, 0].set_xlabel('Example Number')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=np.mean(recon_loss_values), color='red', linestyle='--', 
                      label=f'Avg: {np.mean(recon_loss_values):.6f}')
    axes[1, 0].legend()
    
    # KL Divergence plot
    axes[1, 1].bar(range(1, len(kl_loss_values) + 1), kl_loss_values, color='purple', alpha=0.7)
    axes[1, 1].set_title('KL Divergence Loss')
    axes[1, 1].set_xlabel('Example Number')
    axes[1, 1].set_ylabel('KL Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=np.mean(kl_loss_values), color='red', linestyle='--', 
                      label=f'Avg: {np.mean(kl_loss_values):.6f}')
    axes[1, 1].legend()
    
    # Combined loss plot
    total_loss_values = [m['total_loss'] for m in all_metrics]
    axes[1, 2].bar(range(1, len(total_loss_values) + 1), total_loss_values, color='gray', alpha=0.7)
    axes[1, 2].set_title('Total VAE Loss')
    axes[1, 2].set_xlabel('Example Number')
    axes[1, 2].set_ylabel('Total Loss')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=np.mean(total_loss_values), color='red', linestyle='--', 
                      label=f'Avg: {np.mean(total_loss_values):.6f}')
    axes[1, 2].legend()
    
    plt.suptitle('VAE Reconstruction Metrics Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'vae_summary_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\nVAE Reconstruction Summary Statistics:")
    print(f"Average MSE: {np.mean(mse_values):.6f} ± {np.std(mse_values):.6f}")
    print(f"Average MAE: {np.mean(mae_values):.6f} ± {np.std(mae_values):.6f}")
    print(f"Average PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f} dB")
    print(f"Average Reconstruction Loss: {np.mean(recon_loss_values):.6f} ± {np.std(recon_loss_values):.6f}")
    print(f"Average KL Divergence: {np.mean(kl_loss_values):.6f} ± {np.std(kl_loss_values):.6f}")
    print(f"Average Total Loss: {np.mean(total_loss_values):.6f} ± {np.std(total_loss_values):.6f}")


def visualize_vae_results(args):
    """Main VAE visualization function"""
    
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
    
    # Load dataset (use image mode for VAE)
    print("Loading dataset...")
    dataset = HASCID_data(
        data_path=args.data_path,
        train_mode='image',  # Force image mode for VAE
        split=args.split,
        eval_ratio=args.eval_ratio,
        data_format='image'  # Force image format for VAE
    )
    
    # Create dataloader with image collate function
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, 
                          collate_fn=image_collate_fn)
    
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # Load VAE model
    model = load_vae_model(args.model_path, device, args.spectral_channels, 
                          args.latent_channels, args.base_channels)
    
    # Get wavelengths from dataset (first 160 channels used as GT)
    wavelengths = dataset.wavelens[:args.spectral_channels]
    
    print(f"Generating VAE visualizations for {args.num_examples} examples...")
    
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
        
        print(f"\nProcessing example {examples_processed + 1}")
        print(f"GT data shape: {gt_data.shape}")
        print(f"GT data range: [{gt_data.min():.3f}, {gt_data.max():.3f}]")
        
        # Reconstruct with VAE
        reconstructed_data, mean, logvar, loss_metrics = reconstruct_with_vae(
            model, gt_data, device, sample=True  # Use mean for visualization
        )
        
        print(f"Reconstructed data shape: {reconstructed_data.shape}")
        print(f"Reconstructed data range: [{reconstructed_data.min():.3f}, {reconstructed_data.max():.3f}]")
        print(f"Latent mean shape: {mean.shape}, Latent logvar shape: {logvar.shape}")
        
        # Create comparison plot
        example_metrics = create_vae_comparison_plot(
            gt_data, reconstructed_data, wavelengths, examples_processed, 
            args.save_path, loss_metrics, args.spatial_region
        )
        all_metrics.append(example_metrics)
        
        # Create latent space analysis
        latent_stats = create_latent_analysis_plot(
            mean, logvar, examples_processed, args.save_path
        )
        all_latent_stats.append(latent_stats)
        
        examples_processed += 1
        
        print(f"Example {examples_processed} metrics:")
        print(f"  MSE: {example_metrics['mse']:.6f}")
        print(f"  MAE: {example_metrics['mae']:.6f}")
        print(f"  PSNR: {example_metrics['psnr']:.2f} dB")
        print(f"  Recon Loss: {example_metrics['recon_loss']:.6f}")
        print(f"  KL Loss: {example_metrics['kl_loss']:.6f}")
    
    # Create summary plot
    create_summary_plot(all_metrics, args.save_path)
    
    # Save detailed metrics to file
    metrics_file = os.path.join(args.save_path, 'vae_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Example\tMSE\tMAE\tPSNR\tRecon_Loss\tKL_Loss\tTotal_Loss\tSpectral_MSE\tSpectral_MAE\n")
        for i, metrics in enumerate(all_metrics):
            f.write(f"{i+1}\t{metrics['mse']:.6f}\t{metrics['mae']:.6f}\t{metrics['psnr']:.2f}\t"
                   f"{metrics['recon_loss']:.6f}\t{metrics['kl_loss']:.6f}\t{metrics['total_loss']:.6f}\t"
                   f"{metrics['spectral_mse']:.6f}\t{metrics['spectral_mae']:.6f}\n")
        
        # Add averages
        f.write(f"\nAverage\t{np.mean([m['mse'] for m in all_metrics]):.6f}\t"
               f"{np.mean([m['mae'] for m in all_metrics]):.6f}\t"
               f"{np.mean([m['psnr'] for m in all_metrics]):.2f}\t"
               f"{np.mean([m['recon_loss'] for m in all_metrics]):.6f}\t"
               f"{np.mean([m['kl_loss'] for m in all_metrics]):.6f}\t"
               f"{np.mean([m['total_loss'] for m in all_metrics]):.6f}\t"
               f"{np.mean([m['spectral_mse'] for m in all_metrics]):.6f}\t"
               f"{np.mean([m['spectral_mae'] for m in all_metrics]):.6f}\n")
    
    # Save latent space statistics
    latent_file = os.path.join(args.save_path, 'latent_stats.txt')
    with open(latent_file, 'w') as f:
        f.write("Example\tLatent_Mean_Norm\tLatent_Std_Mean\tCompression_Ratio\n")
        for i, stats in enumerate(all_latent_stats):
            f.write(f"{i+1}\t{stats['latent_mean_norm']:.6f}\t{stats['latent_std_mean']:.6f}\t"
                   f"{stats['latent_compression_ratio']:.2f}\n")
    
    print(f"\nVAE visualization complete! Results saved to: {args.save_path}")
    print(f"- Individual comparison plots: vae_comparison_example_*.png")
    print(f"- Latent analysis plots: latent_analysis_example_*.png")
    print(f"- Summary metrics plot: vae_summary_metrics.png")
    print(f"- Detailed metrics: vae_metrics.txt")
    print(f"- Latent statistics: latent_stats.txt")


def check_latent_range(args):
    """Check the range of latent variables across the dataset"""
    model = load_vae_model(args.model_path, torch.device(args.device), args.spectral_channels, 
                          args.latent_channels, args.base_channels)
    dataset = HASCID_data(
        data_path=args.data_path,
        train_mode='image',
        split='train',  # Use training set for range check
        eval_ratio=args.eval_ratio,
        data_format='image'
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, 
                          collate_fn=image_collate_fn)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    all_means = []
    all_logvars = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Checking latent range"):
            gt_data, _ = batch
            gt_data = gt_data.to(device)
            
            _, mean, logvar = model(gt_data, sample=False)
            all_means.append(mean.cpu())
            all_logvars.append(logvar.cpu())
    
    all_means = torch.cat(all_means, dim=0)  # [N, latent_channels, H', W']
    all_logvars = torch.cat(all_logvars, dim=0)
    
    mean_min = all_means.min().item()
    mean_max = all_means.max().item()
    logvar_min = all_logvars.min().item()
    logvar_max = all_logvars.max().item()
    
    print(f"Latent Mean Range: [{mean_min:.3f}, {mean_max:.3f}]")
    print(f"Latent LogVar Range: [{logvar_min:.3f}, {logvar_max:.3f}]")
    
    return (mean_min, mean_max), (logvar_min, logvar_max)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    print("VAE Visualization Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    # check_latent_range(args)
    visualize_vae_results(args)
