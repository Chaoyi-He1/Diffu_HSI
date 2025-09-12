# 1D U-Net Diffusion Training Utilities

This directory contains training utilities for 1D U-Net diffusion models. The main `train.py` file provides modular functions that take model, trainer, dataloader etc. as inputs.

## Main Functions

### `train_one_epoch`

The core training function that trains a model for one epoch.

**Parameters:**

- `model`: The 1D U-Net model to train
- `diffusion_trainer`: Diffusion trainer with loss computation
- `dataloader`: Training data loader yielding (data, conditions) tuples
- `optimizer`: PyTorch optimizer
- `epoch`: Current epoch number
- `device`: Device to run training on
- `grad_clip`: Gradient clipping threshold (optional)
- `log_interval`: Log metrics every N steps
- `logger`: Optional logger for detailed logging

**Returns:**

Dictionary with metrics: loss, learning_rate, samples_per_sec, grad_norm, total_samples

### Other Utilities

- `validate_one_epoch`: Validation for one epoch
- `generate_samples`: Generate samples using trained model
- `train_full_pipeline`: Complete training pipeline
- `save_checkpoint`/`load_checkpoint`: Checkpoint utilities
- `create_synthetic_1d_dataset`: Create synthetic data for testing

## Usage Example

```python
from train import train_one_epoch
from model.u2net_1d import U2Net1D
from model.diffusion_trainer import DiffusionTrainer

# Setup model, trainer, dataloader, optimizer
model = U2Net1D(input_channels=3, condition_dim=64, base_channels=32)
diffusion_trainer = DiffusionTrainer(n_timesteps=1000, prediction_type="eps")
# ... setup dataloader and optimizer ...

# Train for one epoch
metrics = train_one_epoch(
    model=model,
    diffusion_trainer=diffusion_trainer,
    dataloader=train_dataloader,
    optimizer=optimizer,
    epoch=0,
    device=device,
    grad_clip=1.0
)

print(f"Training loss: {metrics['loss']:.6f}")
```

## Files

- `train.py`: Main training utilities with `train_one_epoch` function
- `example_usage.py`: Complete example showing how to use the training functions
- `README.md`: This documentation

## Running the Example

```bash
cd train_eval
python example_usage.py
```

This will run a complete training example with synthetic data, demonstrating how to use all the training utilities.
