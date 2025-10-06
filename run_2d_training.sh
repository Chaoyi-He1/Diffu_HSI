#!/bin/bash

# Training script for 2D hyperspectral diffusion without VAE
# This script runs the direct 2D diffusion training

echo "Starting 2D Hyperspectral Diffusion Training (Direct, without VAE)"
echo "================================================================="

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
fi

# Default parameters optimized for direct training
SPECTRAL_CHANNELS=160
SENSOR_CHANNELS=30
BASE_CHANNELS=64
BATCH_SIZE=2
LR=5e-5
NUM_EPOCHS=1000

# You can override these by setting environment variables
SPECTRAL_CHANNELS=${SPECTRAL_CHANNELS:-160}
SENSOR_CHANNELS=${SENSOR_CHANNELS:-30}
BASE_CHANNELS=${BASE_CHANNELS:-64}
BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-5e-5}
NUM_EPOCHS=${NUM_EPOCHS:-1000}

echo "Training Configuration:"
echo "  Spectral channels: $SPECTRAL_CHANNELS"
echo "  Sensor channels: $SENSOR_CHANNELS"
echo "  Base channels: $BASE_CHANNELS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Number of epochs: $NUM_EPOCHS"
echo ""

# Create results directory
mkdir -p results/2d_hsi_diffusion

# Run training
python main_2d.py \
    --data_path dataset/HASCID-Dataset \
    --train_mode image \
    --spectral_channels $SPECTRAL_CHANNELS \
    --sensor_channels $SENSOR_CHANNELS \
    --base_channels $BASE_CHANNELS \
    --batch_size $BATCH_SIZE \
    --num_workers 4 \
    --lr $LR \
    --num_epochs $NUM_EPOCHS \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --scaler amp \
    --val_every 100 \
    --save_every 50 \
    --generate_every 200 \
    --log_interval 50 \
    --loss_type l2 \
    --noise_schedule linear \
    --timesteps 1000 \
    --prediction_type eps \
    --save_path results/2d_hsi_diffusion \
    --seed 42 \
    --device cuda

echo ""
echo "Training completed!"
echo "Results saved in: results/2d_hsi_diffusion"
echo ""
echo "To resume training from a checkpoint, use:"
echo "  python main_2d.py --resume results/2d_hsi_diffusion/checkpoint_epoch_XXX.pth [other args]"
echo ""
echo "To reduce memory usage if you encounter OOM errors:"
echo "  BATCH_SIZE=1 BASE_CHANNELS=32 ./run_2d_training.sh"