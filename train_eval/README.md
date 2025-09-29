# Hyperspectral Diffusion Training Utilities

This directory contains comprehensive training utilities for hyperspectral diffusion models, including 1D U-Net, VAE, and 2D latent diffusion approaches.

## Available Training Scripts

### 1. `train_1d.py` - 1D U-Net Diffusion Training

Training utilities for 1D U-Net diffusion models that process hyperspectral data as 1D sequences.

**Key Functions:**
- `train_one_epoch`: Core training function for one epoch
- `validate_one_epoch`: Validation for one epoch  
- `generate_samples`: Generate samples using trained model
- `train_full_pipeline`: Complete training pipeline
- `save_checkpoint`/`load_checkpoint`: Checkpoint utilities

### 2. `train_vae.py` - VAE Training

Comprehensive training utilities for Hyperspectral VAE models that learn compressed latent representations.

**Key Functions:**
- `train_vae_full`: Complete VAE training pipeline
- `train_vae_one_epoch`: Train VAE for one epoch
- `validate_vae_one_epoch`: Validate VAE for one epoch
- `compute_vae_metrics`: Comprehensive VAE metrics (PSNR, spectral metrics, etc.)
- `generate_vae_samples`: Generate VAE reconstruction samples

### 3. `train_2d_with_vae.py` - 2D Latent Diffusion Training ⭐ NEW

**Complete 2D latent diffusion training pipeline** that combines VAE and diffusion in latent space:

**Core Concept:**

1. **VAE Encoding**: Compress hyperspectral images from (B, 128, 256, 256) to (B, 8, 32, 32)
2. **Latent Diffusion**: Train diffusion model in this compressed latent space
3. **VAE Decoding**: Decode generated latents back to full hyperspectral images

**Key Features:**

- **Extreme Memory Efficiency**: ~1000x compression ratio (operates on 8×32×32 instead of 128×256×256)
- **Quality Preservation**: Pre-trained VAE ensures high-fidelity reconstructions
- **Pretrained VAE Support**: Load existing VAE checkpoints for faster training
- **Comprehensive Pipeline**: Includes training, validation, sampling, and checkpointing
- **Multi-Resolution Context**: Handles sensor conditioning at multiple scales

**Main Class:**

```python
from train_2d_with_vae import LatentDiffusion2DTrainer, create_default_config

config = create_default_config()
config.update({
    'data_dir': 'path/to/HASCID-Dataset',
    'vae_checkpoint_path': 'path/to/pretrained_vae.pth',
    'batch_size': 16,  # Can be larger due to latent compression
    'epochs': 1000
})

trainer = LatentDiffusion2DTrainer(config)
trainer.train()
```

## Usage Examples

### Quick Start - 2D Latent Diffusion

```python
# Example 1: Train from scratch (includes VAE training)
from train_2d_with_vae import train_2d_with_vae_full

trained_model = train_2d_with_vae_full(
    data_path="/path/to/HASCID-Dataset",
    save_dir="./results/2d_latent_diffusion",
    batch_size=4,
    vae_config={'num_epochs': 50},
    diffusion_config={'num_epochs': 100}
)
```

```python
# Example 2: Use pretrained VAE
trained_model = train_2d_with_vae_full(
    data_path="/path/to/HASCID-Dataset", 
    save_dir="./results/2d_with_pretrained_vae",
    pretrained_vae_path="./results/vae_best_model.pth",
    skip_vae_training=True,
    diffusion_config={'num_epochs': 200}
)
```

### 1D Diffusion Example

```python
from train_1d import train_one_epoch
from model.u2net_1d import U2Net1D
from model.diffusion_trainer import DiffusionTrainer

model = U2Net1D(input_channels=3, condition_dim=64, base_channels=32)
diffusion_trainer = DiffusionTrainer(n_timesteps=1000, prediction_type="eps")

metrics = train_one_epoch(
    model=model,
    diffusion_trainer=diffusion_trainer,
    dataloader=train_dataloader,
    optimizer=optimizer,
    epoch=0,
    device=device,
    grad_clip=1.0
)
```

### VAE Training Example

```python
from train_vae import train_vae_full
from model.hyperspectral_vae import HyperspectralVAE

vae = HyperspectralVAE(spectral_channels=160, latent_channels=8)

trained_vae, history = train_vae_full(
    model=vae,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_epochs=100,
    device=device,
    save_dir="./vae_checkpoints"
)
```

## File Structure

```bash
train_eval/
├── train_1d.py              # 1D diffusion training utilities
├── train_vae.py             # VAE training utilities  
├── train_2d_with_vae.py     # 2D latent diffusion training (NEW)
├── example_usage.py         # 1D diffusion examples
├── example_2d_usage.py      # 2D latent diffusion examples (NEW)
├── train_simple_example.py  # Simple training example
└── README.md               # This documentation
```

## Running Examples

```bash
# Run 2D latent diffusion examples
cd train_eval
python example_2d_usage.py

# Run 1D diffusion examples  
python example_usage.py

# Run simple training example
python train_simple_example.py
```

## Model Comparison

| Approach | Input Resolution | Processing | Memory Usage | Training Speed |
|----------|------------------|------------|--------------|----------------|
| 1D Diffusion | 1D sequences | Direct | Medium | Fast |
| VAE Only | 2D images | Autoencoder | Low | Very Fast |
| **2D Latent Diffusion** | **2D images** | **VAE + Diffusion** | **Very Low** | **Medium** |

## Key Advantages of 2D Latent Diffusion

1. **Memory Efficiency**: ~1000x compression (512×512×128 → 64×64×8)
2. **Quality Preservation**: VAE maintains spectral fidelity
3. **Scalability**: Can handle large hyperspectral images
4. **Flexibility**: Supports pretrained VAE models
5. **Comprehensive**: Includes both VAE and diffusion training

## Configuration Options

The 2D latent diffusion training supports extensive configuration:

```python
vae_config = {
    'num_epochs': 100,
    'learning_rate': 1e-4, 
    'kl_weight': 1e-6,     # KL divergence weight
    'grad_clip': 1.0,
    'val_every': 5,
    'save_every': 10,
    'generate_every': 20
}

diffusion_config = {
    'num_epochs': 200,
    'learning_rate': 2e-4,
    'n_timesteps': 1000,
    'beta_start': 1e-4,
    'beta_end': 0.02,
    'prediction_type': 'eps',  # 'eps', 'x0', or 'v'
    'loss_type': 'l2',         # 'l1' or 'l2'
    'val_every': 10,
    'save_every': 20,
    'generate_every': 50
}
```

## Getting Started

1. **For new users**: Start with `train_2d_with_vae.py` for state-of-the-art results
2. **For memory-constrained setups**: Use smaller batch sizes and model configurations
3. **For experimentation**: Use the provided example scripts as templates
4. **For production**: Configure appropriate validation and checkpointing intervals
