"""
Visualization for the channel–spatial (Conv3d-style) U-Net variant trained with main_2d_fsdp.py.

I/O tensors are still BCHW hyperspectral cubes; only the internal conv blocks differ from the
plain 2D visual script. Defaults align with --use_channel_3d_conv FSDP runs.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from torch.utils.data import DataLoader
import sys
import random
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader.my_dataset import HASCID_data, image_collate_fn
from data_loader.HFD_dataset import HFD_data
from model.u2net_hyperspectral import U2NetHyperspectral
from model.diffusion_trainer import DiffusionTrainer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Visualization for channel-spatial (3D conv block) HSI diffusion (same BCHW I/O as 2D script)'
    )

    parser.add_argument('--data_path', type=str, default='dataset/HFD100 Mat dataset',
                        choices=['dataset/HASCID-Dataset', 'dataset/HFD100 Mat dataset'],
                        help='path to dataset')
    parser.add_argument('--train_mode', type=str, default='image', choices=['pixel', 'image'],
                        help='training mode')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to load data')
    parser.add_argument('--eval_ratio', type=float, default=0.1, help='the ratio of test data during training')
    parser.add_argument('--R-n', type=int, default=1,
                        help='the number of random measurements, if None, then use full measurements')
    parser.add_argument('--dataset', type=str, choices=['HASCID', 'HFD'], default='HFD', help='which dataset to use')
    parser.add_argument('--ds', '--sensor_down_sample_rate', type=int, default=2,
                        help='down sample rate for sensor response when train_mode is image',
                        dest='sensor_down_sample_rate')

    parser.add_argument('--spectral_channels', type=int, default=64, help='spectral bands of HSI')
    parser.add_argument('--sensor_channels', type=int, default=30, help='sensor response channels')
    parser.add_argument('--base_channels', type=int, default=512, help='base channels of model')
    parser.add_argument('--channel_kernel', type=int, default=7,
                        help='kernel along spectral axis in channel-spatial conv (match training)')
    parser.add_argument('--spatial_kernel', type=int, default=3,
                        help='spatial H/W kernel in channel-spatial conv (match training)')

    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'l2', 'huber'],
                        help='loss type used by diffusion trainer')
    parser.add_argument('--noise_schedule', type=str, default='linear', choices=['linear', 'cosine'],
                        help='noise schedule for diffusion process')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of diffusion timesteps')
    parser.add_argument('--prediction_type', type=str, default='eps', choices=['eps', 'x0', 'v'],
                        help='diffusion prediction type')

    parser.add_argument('--num_examples', type=int, default=5, help='number of examples to visualize')
    parser.add_argument('--model_path', '--resume', type=str,
                        default='results/2d_hsi_diffusion/down_sample_2/HFD/R_1/l1_loss_fsdp_3dconv/checkpoint_epoch_40.pth',
                        dest='model_path', help='path to trained model checkpoint')
    parser.add_argument('--save_path', type=str,
                        default='results/3d_conv_visualization/down_sample_2/HFD/R_1/l1_loss_fsdp_3dconv',
                        help='path to save visualization results')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='dataset split to visualize')

    parser.add_argument('--num_steps', type=int, default=1000, help='number of denoising steps')
    parser.add_argument('--method', type=str, default='ddpm', choices=['ddpm', 'ddim'], help='sampling method')

    parser.add_argument('--spatial_region', type=str, default='center', choices=['center', 'random', 'corner'],
                        help='which spatial region to analyze in detail')
    parser.add_argument('--num_spectral_bands', type=int, default=6, help='number of spectral bands to visualize')

    parser.add_argument('--device', type=str, default='cuda', help='device to use for computation')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    return parser


def load_model(
    model_path: str,
    device: torch.device,
    spectral_channels: int = 64,
    sensor_channels: int = 30,
    base_channels: int = 512,
    channel_kernel: int = 7,
    spatial_kernel: int = 3,
):
    """Load the trained U-Net with channel-spatial (3D) conv blocks."""
    model = U2NetHyperspectral(
        spectral_channels=spectral_channels,
        sensor_channels=sensor_channels,
        base_channels=base_channels,
        use_channel_3d_conv=True,
        channel_kernel=channel_kernel,
        spatial_kernel=spatial_kernel,
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    for k, v in model.named_parameters():
        if k not in state_dict or not torch.equal(v, state_dict[k]):
            print(f"Warning: Parameter {k} does not match checkpoint value.")
            raise ValueError("Model parameters do not match checkpoint.")
    model = model.to(device)
    model.eval()
    print("Model loaded successfully (use_channel_3d_conv=True)")
    return model


def generate_samples(model, diffusion_trainer, gt_data, sensor_data, device, num_steps=50, method='ddpm'):
    model.eval()
    with torch.no_grad():
        batch_size, spectral_channels, H, W = gt_data.shape
        generated = diffusion_trainer.sample(
            model=model,
            cond=sensor_data,
            shape=(batch_size, spectral_channels, H, W),
            n_steps=num_steps,
            method=method,
            progress=True,
        )
    return generated


def create_comparison_plot(gt_data, sensor_data, generated_data, wavelengths,
                           example_idx, save_path, spatial_region='center'):
    os.makedirs(save_path, exist_ok=True)

    B, C, _, _ = gt_data.shape
    example_metrics = []
    for idx in range(B):
        gt_img = gt_data[idx].cpu().numpy()
        sensor_img = sensor_data[idx].cpu().numpy()
        generated_img = generated_data[idx].cpu().numpy()

        if len(wavelengths) >= 3:
            rgb_indices = [
                np.argmin(np.abs(wavelengths - 650)),
                np.argmin(np.abs(wavelengths - 550)),
                np.argmin(np.abs(wavelengths - 450)),
            ]
        else:
            rgb_indices = [0, C // 2, C - 1]

        gt_rgb = np.stack([gt_img[rgb_indices[0]], gt_img[rgb_indices[1]], gt_img[rgb_indices[2]]], axis=-1)
        generated_rgb = np.stack(
            [generated_img[rgb_indices[0]], generated_img[rgb_indices[1]], generated_img[rgb_indices[2]]], axis=-1
        )

        gt_rgb = np.clip((gt_rgb - np.min(gt_rgb)) / (np.max(gt_rgb) - np.min(gt_rgb)), 0, 1)
        generated_rgb = np.clip(
            (generated_rgb - np.min(generated_rgb)) / (np.max(generated_rgb) - np.min(generated_rgb)), 0, 1
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f'Example {example_idx + 1 + idx}: HSI reconstruction (channel–spatial 3D conv U-Net)',
            fontsize=16,
        )

        axes[0].imshow(gt_rgb)
        axes[0].set_title('Ground Truth (RGB)')
        axes[0].axis('off')

        axes[1].imshow(generated_rgb)
        axes[1].set_title('Generated (RGB)')
        axes[1].axis('off')

        diff_rgb = np.abs(gt_rgb - generated_rgb)
        axes[2].imshow(diff_rgb)
        axes[2].set_title('Absolute Difference (RGB)')
        axes[2].axis('off')

        plt.tight_layout()
        plot_path = os.path.join(save_path, f'comparison_3dconv_example_{example_idx + 1 + idx}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        gt_img_batch = np.expand_dims(gt_img, axis=0)
        generated_img_batch = np.expand_dims(generated_img, axis=0)

        create_spectral_analysis_plot(
            gt_img_batch, generated_img_batch, wavelengths, example_idx + idx, save_path, spatial_region
        )

        mse = np.mean((gt_img - generated_img) ** 2)
        mae = np.mean(np.abs(gt_img - generated_img))
        psnr = 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')

        spectral_mse = np.mean(np.mean((gt_img - generated_img) ** 2, axis=(1, 2)))
        spectral_mae = np.mean(np.mean(np.abs(gt_img - generated_img), axis=(1, 2)))

        example_metrics.append({
            'mse': mse,
            'mae': mae,
            'psnr': psnr,
            'spectral_mse': spectral_mse,
            'spectral_mae': spectral_mae,
        })

    return example_metrics


def create_spectral_analysis_plot(gt_img, generated_img, wavelengths, example_idx, save_path, spatial_region='center'):
    B, C, H, W = gt_img.shape
    region_size = (H, W)

    if spatial_region == 'center':
        h_start = max(0, (H - region_size[0]) // 2)
        w_start = max(0, (W - region_size[1]) // 2)
    elif spatial_region == 'corner':
        h_start, w_start = 0, 0
    else:
        h_start = random.randint(0, max(0, H - region_size[0]))
        w_start = random.randint(0, max(0, W - region_size[1]))

    h_end = min(H, h_start + region_size[0])
    w_end = min(W, w_start + region_size[1])

    for b_idx in range(B):
        sample_pixels = [
            (h_start + i, w_start + j)
            for i in range(0, h_end - h_start, max(1, (h_end - h_start) // 4))
            for j in range(0, w_end - w_start, max(1, (w_end - w_start) // 4))
        ][:9]

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(
            f'Example {example_idx + 1 + b_idx}: Spectral profiles (3D conv U-Net)',
            fontsize=16,
        )

        for idx, (h, w) in enumerate(sample_pixels):
            if idx >= 9:
                break
            row, col = idx // 3, idx % 3
            ax = axes[row, col]

            gt_spectrum = gt_img[b_idx, :, h, w]
            generated_spectrum = generated_img[b_idx, :, h, w]

            ax.plot(wavelengths, gt_spectrum, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
            ax.plot(wavelengths, generated_spectrum, 'r--', linewidth=2, label='Generated', alpha=0.8)

            ax.set_title(f'Pixel ({h}, {w})')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Reflectance')
            ax.grid(True, alpha=0.3)
            ax.legend()

            pixel_mse = np.mean((gt_spectrum - generated_spectrum) ** 2)
            pixel_mae = np.mean(np.abs(gt_spectrum - generated_spectrum))
            metrics_text = f'MSE: {pixel_mse:.6f}\nMAE: {pixel_mae:.6f}'
            ax.text(
                0.02, 0.98, metrics_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            )

        for idx in range(len(sample_pixels), 9):
            axes[idx // 3, idx % 3].axis('off')

        plt.tight_layout()
        spectral_analysis_path = os.path.join(save_path, f'spectral_analysis_3dconv_example_{example_idx + 1 + b_idx}.png')
        plt.savefig(spectral_analysis_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_summary_plot(all_metrics, save_path):
    if not all_metrics:
        print("No metrics available; skipping summary plot.")
        return

    mse_values = [m['mse'] for m in all_metrics]
    mae_values = [m['mae'] for m in all_metrics]
    psnr_values = [m['psnr'] for m in all_metrics]
    spectral_mse_values = [m['spectral_mse'] for m in all_metrics]
    spectral_mae_values = [m['spectral_mae'] for m in all_metrics]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Channel–spatial (3D conv) HSI diffusion — performance summary', fontsize=16)

    axes[0, 0].bar(range(1, len(mse_values) + 1), mse_values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Mean Squared Error (MSE)')
    axes[0, 0].set_xlabel('Example Number')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=np.mean(mse_values), color='red', linestyle='--', label=f'Average: {np.mean(mse_values):.6f}')
    axes[0, 0].legend()

    axes[0, 1].bar(range(1, len(mae_values) + 1), mae_values, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Mean Absolute Error (MAE)')
    axes[0, 1].set_xlabel('Example Number')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=np.mean(mae_values), color='red', linestyle='--', label=f'Average: {np.mean(mae_values):.6f}')
    axes[0, 1].legend()

    axes[0, 2].bar(range(1, len(psnr_values) + 1), psnr_values, color='lightgreen', alpha=0.7)
    axes[0, 2].set_title('Peak Signal-to-Noise Ratio (PSNR)')
    axes[0, 2].set_xlabel('Example Number')
    axes[0, 2].set_ylabel('PSNR (dB)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=np.mean(psnr_values), color='red', linestyle='--', label=f'Average: {np.mean(psnr_values):.2f} dB')
    axes[0, 2].legend()

    axes[1, 0].bar(range(1, len(spectral_mse_values) + 1), spectral_mse_values, color='orange', alpha=0.7)
    axes[1, 0].set_title('Spectral MSE')
    axes[1, 0].set_xlabel('Example Number')
    axes[1, 0].set_ylabel('Spectral MSE')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(
        y=np.mean(spectral_mse_values), color='red', linestyle='--',
        label=f'Average: {np.mean(spectral_mse_values):.6f}',
    )
    axes[1, 0].legend()

    axes[1, 1].bar(range(1, len(spectral_mae_values) + 1), spectral_mae_values, color='purple', alpha=0.7)
    axes[1, 1].set_title('Spectral MAE')
    axes[1, 1].set_xlabel('Example Number')
    axes[1, 1].set_ylabel('Spectral MAE')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(
        y=np.mean(spectral_mae_values), color='red', linestyle='--',
        label=f'Average: {np.mean(spectral_mae_values):.6f}',
    )
    axes[1, 1].legend()

    axes[1, 2].plot(range(1, len(mse_values) + 1), mse_values, 'o-', label='MSE', alpha=0.7)
    axes[1, 2].plot(range(1, len(mae_values) + 1), mae_values, 's-', label='MAE', alpha=0.7)
    axes[1, 2].set_title('MSE vs MAE Comparison')
    axes[1, 2].set_xlabel('Example Number')
    axes[1, 2].set_ylabel('Error Value')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'summary_metrics_3dconv.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nChannel–spatial (3D conv) HSI diffusion summary statistics:")
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
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    print("Loading dataset...")
    if args.dataset == 'HASCID':
        dataset = HASCID_data(
            data_path=args.data_path,
            train_mode='image',
            split=args.split,
            eval_ratio=args.eval_ratio,
            data_format='image',
            R_n=args.R_n,
            sensor_down_sample_rate=args.sensor_down_sample_rate,
        )
    elif args.dataset == 'HFD':
        dataset = HFD_data(
            data_path=args.data_path,
            train_mode='image',
            split=args.split,
            eval_ratio=args.eval_ratio,
            data_format='image',
            R_n=args.R_n,
            sensor_down_sample_rate=args.sensor_down_sample_rate,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    args.batch_size = min(args.num_examples, 2)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=image_collate_fn,
    )

    print(f"Dataset loaded. Total samples: {len(dataset)}")

    model = load_model(
        args.model_path, device,
        args.spectral_channels, args.sensor_channels, args.base_channels,
        channel_kernel=args.channel_kernel,
        spatial_kernel=args.spatial_kernel,
    )

    diffusion_trainer = DiffusionTrainer(
        n_timesteps=args.timesteps,
        loss_type=args.loss_type,
        prediction_type=args.prediction_type,
        beta_schedule=args.noise_schedule,
        device=device,
    )

    wavelengths = dataset.wavelens[:args.spectral_channels] if args.dataset == 'HASCID' else dataset.new_wavelens[:args.spectral_channels]

    print(f"Generating visualizations for {args.num_examples} examples...")

    all_metrics = []
    examples_processed = 0

    for _batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing examples")):
        if examples_processed >= args.num_examples:
            break

        gt_data, sensor_data = batch
        if examples_processed + gt_data.shape[0] > args.num_examples:
            n = args.num_examples - examples_processed
            gt_data = gt_data[:n]
            sensor_data = sensor_data[:n]

        gt_data = gt_data.to(device)
        sensor_data = sensor_data.to(device)

        print(f"\nProcessing example {examples_processed + 1}")
        print(f"GT data shape: {gt_data.shape}")
        print(f"Sensor data shape: {sensor_data.shape}")

        with torch.no_grad():
            generated_data = generate_samples(
                model, diffusion_trainer, gt_data, sensor_data, device,
                num_steps=args.num_steps, method=args.method,
            )

        batch_metrics = create_comparison_plot(
            gt_data, sensor_data, generated_data,
            wavelengths, examples_processed, args.save_path,
            args.spatial_region,
        )

        all_metrics.extend(batch_metrics)
        examples_processed += gt_data.shape[0]

        for idx, example_metrics in enumerate(batch_metrics):
            print(f"Example {examples_processed - gt_data.shape[0] + idx + 1} metrics:")
            print(f"  MSE: {example_metrics['mse']:.6f}")
            print(f"  MAE: {example_metrics['mae']:.6f}")
            print(f"  PSNR: {example_metrics['psnr']:.2f} dB")
            print(f"  Spectral MSE: {example_metrics['spectral_mse']:.6f}")
            print(f"  Spectral MAE: {example_metrics['spectral_mae']:.6f}")

    create_summary_plot(all_metrics, args.save_path)

    metrics_file = os.path.join(args.save_path, 'metrics_3dconv.txt')
    with open(metrics_file, 'w') as f:
        f.write("Example\tMSE\tMAE\tPSNR\tSpectral_MSE\tSpectral_MAE\n")
        for i, metrics in enumerate(all_metrics):
            f.write(
                f"{i+1}\t{metrics['mse']:.6f}\t{metrics['mae']:.6f}\t{metrics['psnr']:.2f}\t"
                f"{metrics['spectral_mse']:.6f}\t{metrics['spectral_mae']:.6f}\n"
            )
        if all_metrics:
            f.write(
                f"\nAverage\t{np.mean([m['mse'] for m in all_metrics]):.6f}\t"
                f"{np.mean([m['mae'] for m in all_metrics]):.6f}\t"
                f"{np.mean([m['psnr'] for m in all_metrics]):.2f}\t"
                f"{np.mean([m['spectral_mse'] for m in all_metrics]):.6f}\t"
                f"{np.mean([m['spectral_mae'] for m in all_metrics]):.6f}\n"
            )

    print(f"\nVisualization complete. Results saved to: {args.save_path}")
    print("- comparison_3dconv_example_*.png")
    print("- spectral_analysis_3dconv_example_*.png")
    print("- summary_metrics_3dconv.png")
    print("- metrics_3dconv.txt")


def normalize_paths_from_main(args):
    """Align dataset/save path conventions with main_2d_fsdp.py (3D conv run folders)."""
    path_parts = os.path.normpath(args.save_path).split(os.sep)

    if "HASCID" in args.data_path:
        if args.dataset != 'HASCID':
            args.dataset = 'HASCID'
            print("\033[91mWarning: dataset argument changed to 'HASCID' to match data_path.\033[0m", flush=True)
        if len(path_parts) >= 3 and "HASCID" not in path_parts[-3]:
            path_parts[-3] = 'HASCID'
            args.save_path = os.path.join(*path_parts)
            print("\033[91mWarning: save_path argument changed to include 'HASCID' folder.\033[0m", flush=True)
    elif "HFD" in args.data_path:
        if args.dataset != 'HFD':
            args.dataset = 'HFD'
            print("\033[91mWarning: dataset argument changed to 'HFD' to match data_path.\033[0m", flush=True)
        if len(path_parts) >= 3 and "HFD" not in path_parts[-3]:
            path_parts[-3] = 'HFD'
            args.save_path = os.path.join(*path_parts)
            print("\033[91mWarning: save_path argument changed to include 'HFD' folder.\033[0m", flush=True)

    path_parts = os.path.normpath(args.save_path).split(os.sep)
    if args.R_n is None:
        if len(path_parts) >= 2 and "PH5" not in path_parts[-2]:
            path_parts[-2] = 'PH5'
            args.save_path = os.path.join(*path_parts)
            print("\033[91mWarning: save_path argument changed to include 'PH5' folder for full measurements.\033[0m", flush=True)
    elif args.R_n == 1:
        if len(path_parts) >= 2 and "R_1" not in path_parts[-2]:
            path_parts[-2] = 'R_1'
            args.save_path = os.path.join(*path_parts)
            print("\033[91mWarning: save_path argument changed to include 'R_1' folder for R_n=1.\033[0m", flush=True)
    elif args.R_n == 2:
        if len(path_parts) >= 2 and "R_2" not in path_parts[-2]:
            path_parts[-2] = 'R_2'
            args.save_path = os.path.join(*path_parts)
            print("\033[91mWarning: save_path argument changed to include 'R_2' folder for R_n=2.\033[0m", flush=True)
    else:
        raise ValueError(f"Unsupported R_n value: {args.R_n}. Supported values are 1, 2, or None for full measurements.")

    tail = os.path.normpath(args.save_path).split(os.sep)[-1]
    if args.loss_type == 'l1':
        expected = 'l1_loss_fsdp_3dconv'
        if expected not in tail:
            args.save_path = os.path.join(*os.path.normpath(args.save_path).split(os.sep)[:-1], expected)
            print(f"\033[91mWarning: save_path terminal folder set to '{expected}' for L1 + 3D conv runs.\033[0m", flush=True)
    elif args.loss_type == 'l2':
        expected = 'l2_loss_fsdp_3dconv'
        if expected not in tail:
            args.save_path = os.path.join(*os.path.normpath(args.save_path).split(os.sep)[:-1], expected)
            print(f"\033[91mWarning: save_path terminal folder set to '{expected}' for L2 + 3D conv runs.\033[0m", flush=True)
    elif args.loss_type == 'huber':
        expected = 'huber_loss_fsdp_3dconv'
        if expected not in tail:
            args.save_path = os.path.join(*os.path.normpath(args.save_path).split(os.sep)[:-1], expected)
            print(f"\033[91mWarning: save_path terminal folder set to '{expected}' for Huber + 3D conv runs.\033[0m", flush=True)

    if args.sensor_down_sample_rate > 1:
        parts = os.path.normpath(args.save_path).split(os.sep)
        expected = f'down_sample_{args.sensor_down_sample_rate}'
        if len(parts) >= 4 and expected not in parts[-4]:
            parts[-4] = expected
            args.save_path = os.path.join(*parts)
            print(f"\033[91mWarning: save_path argument changed to include '{expected}' folder.\033[0m", flush=True)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    normalize_paths_from_main(args)

    print("Channel–spatial (3D conv) HSI diffusion visualization arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    visualize_results(args)
