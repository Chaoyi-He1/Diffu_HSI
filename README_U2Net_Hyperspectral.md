# U2Net Hyperspectral Diffusion Model

This repository contains a U2Net-based diffusion model specifically designed for hyperspectral image generation, taking inspiration from Stable Diffusion's architecture and adapting it for the unique challenges of hyperspectral imaging.

## Architecture Overview

The `U2NetHyperspectral` model is designed to generate hyperspectral images with the following characteristics:

- **Input (x_t)**: Noisy hyperspectral image of shape `(B, L, H, W)`
  - `B`: Batch size
  - `L`: Number of spectral channels (e.g., 128 bands)
  - `H, W`: Spatial dimensions (height, width)

- **Context**: Sensor response conditioning of shape `(B, S, H, W)`
  - `S`: Number of sensor response channels (e.g., 16 channels)
  - Provides spatial conditioning information for each pixel

- **Time embedding**: Diffusion timestep `t` of shape `(B,)`
  - Used for denoising schedule in diffusion process

- **Output**: Predicted noise or clean hyperspectral image of shape `(B, L, H, W)`

## Key Features

### 1. **Nested U2Net Architecture**
- Each stage contains a nested U-Net block (`U2NetBlock2D`)
- Multi-scale feature processing within each stage
- Progressive spatial downsampling: H×W → H/2×W/2 → H/4×W/4 → H/8×W/8
- Progressive channel expansion: Base_C → Base_C×2 → Base_C×4 → Base_C×8

### 2. **Cross-Attention Conditioning**
- Transformer-based attention blocks in each stage
- Context (sensor response) is processed at multiple resolutions
- Time embedding combined with spatial context
- Enables fine-grained control over generation process

### 3. **Multi-Resolution Context Handling**
- Context is adaptively downsampled to match each processing stage
- Maintains spatial correspondence between input and conditioning
- Efficient memory usage and computational complexity

### 4. **Skip Connections**
- U-Net style skip connections preserve fine spatial details
- Multi-scale feature fusion at each resolution level
- Robust gradient flow through the network

## Model Components

### Core Classes

1. **`U2NetHyperspectral`**: Main model class
   - Handles overall architecture and forward pass
   - Manages context encoding and time embedding
   - Coordinates multi-resolution processing

2. **`U2NetBlock2D`**: Nested U-Net block for 2D processing
   - Mini U-Net within each stage
   - Cross-attention with context conditioning
   - Spatial feature processing at multiple scales

3. **`ConvBlock2D`**: Basic 2D convolution block
   - Double convolution with group normalization
   - GELU activation for better gradient flow
   - Maintains spatial dimensions with padding

### Supporting Components (from `layers.py`)

- **`BasicTransformerBlock`**: Transformer attention implementation
- **`CrossAttention`**: Cross-attention mechanism
- **`TimeEmbedding`**: Sinusoidal time embedding
- **`FeedForward`**: MLP with gating

## Usage Examples

### Basic Usage

```python
import torch
from u2net_hyperspectral import U2NetHyperspectral

# Model configuration
spectral_channels = 128  # Number of hyperspectral bands
sensor_channels = 16     # Number of sensor response channels
base_channels = 64       # Base channel dimension (controls capacity)

# Create model
model = U2NetHyperspectral(spectral_channels, sensor_channels, base_channels)

# Example input data
batch_size = 4
height, width = 64, 64

x_t = torch.randn(batch_size, spectral_channels, height, width)  # Noisy hyperspectral
context = torch.randn(batch_size, sensor_channels, height, width)  # Sensor response
t = torch.randint(0, 1000, (batch_size,))  # Diffusion timesteps

# Forward pass
output = model(x_t, t, context)
print(f"Output shape: {output.shape}")  # [4, 128, 64, 64]
```

### Training Integration

```python
# In a diffusion training loop
for batch in dataloader:
    clean_images, sensor_responses = batch  # [B, L, H, W], [B, S, H, W]
    
    # Sample random timesteps
    t = torch.randint(0, num_timesteps, (batch_size,))
    
    # Add noise according to diffusion schedule
    noise = torch.randn_like(clean_images)
    x_t = sqrt_alpha_cumprod[t] * clean_images + sqrt_one_minus_alpha_cumprod[t] * noise
    
    # Predict noise
    predicted_noise = model(x_t, t, sensor_responses)
    
    # Compute loss
    loss = F.mse_loss(predicted_noise, noise)
    loss.backward()
    optimizer.step()
```

### Generation/Inference

```python
# Generate hyperspectral image from sensor response
@torch.no_grad()
def generate_hyperspectral(model, sensor_response, num_steps=50):
    B, S, H, W = sensor_response.shape
    
    # Start with random noise
    x = torch.randn(B, spectral_channels, H, W)
    
    # Reverse diffusion process
    for t in reversed(range(num_steps)):
        t_tensor = torch.full((B,), t, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x, t_tensor, sensor_response)
        
        # Denoise (simplified DDPM step)
        alpha_t = alpha_schedule[t]
        x = (x - predicted_noise * (1 - alpha_t) / sqrt(1 - alpha_cumprod[t])) / sqrt(alpha_t)
    
    return x

# Usage
sensor_input = torch.randn(1, 16, 64, 64)
generated_hyperspectral = generate_hyperspectral(model, sensor_input)
```

## Model Specifications

### Memory and Computational Requirements

- **Parameters**: ~49M (with base_channels=64)
- **Memory**: ~8GB VRAM for training with batch_size=4, 64×64 images
- **Inference**: ~2GB VRAM for single image generation

### Scalability

The model is designed to handle various input sizes:
- Minimum: 32×32 pixels
- Recommended: 64×64 to 128×128 pixels  
- Maximum: Limited by available VRAM

### Hyperparameters

- **base_channels**: Controls model capacity (32, 64, 128)
- **spectral_channels**: Number of hyperspectral bands (typically 50-200)
- **sensor_channels**: Number of sensor channels (typically 8-32)

## Differences from Standard Stable Diffusion

1. **Input Format**: 
   - Standard SD: RGB images (3 channels)
   - This model: Hyperspectral images (50-200+ channels)

2. **Conditioning**:
   - Standard SD: Text embeddings (global)
   - This model: Sensor response maps (spatial)

3. **Architecture**:
   - Standard SD: UNet with ResNet blocks
   - This model: Nested U2Net blocks with multi-scale processing

4. **Attention**:
   - Standard SD: Self and cross-attention with text
   - This model: Multi-resolution cross-attention with spatial context

## Applications

- **Hyperspectral image super-resolution**
- **Sensor fusion and data completion**
- **Synthetic hyperspectral data generation**
- **Noise reduction and denoising**
- **Multi-modal remote sensing**

## Future Extensions

- **Latent space diffusion**: Process in lower-dimensional latent space
- **Conditioning variants**: Support for additional conditioning modalities
- **Temporal modeling**: Extend to hyperspectral video sequences
- **Multi-scale training**: Progressive training at different resolutions

## References

1. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion)
2. Qin et al., "U2-Net: Going deeper with nested U-structure for salient object detection"
3. Ho et al., "Denoising Diffusion Probabilistic Models"
4. Vaswani et al., "Attention is All You Need"
