#!/bin/bash

# Example script to run the 1D HSI diffusion model visualization
# Make sure to update the model_path to point to your trained model checkpoint

python visual_1d.py \
    --data_path "dataset/HASCID-Dataset" \
    --model_path "results/1d_hsi_diffusion/checkpoint_epoch_100.pth" \
    --save_path "results/1d_visualization" \
    --num_examples 10 \
    --split "test" \
    --train_mode "pixel" \
    --num_steps 50 \
    --method "ddpm" \
    --sensor_channels 30 \
    --base_channels 128 \
    --device "cuda" \
    --seed 42

echo "Visualization complete! Check the results in the save_path directory."