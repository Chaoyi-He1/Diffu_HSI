#!/bin/bash
# Test script for VAE visualization
echo "Testing VAE visualization with minimal settings..."

cd /data/chaoyi_he/HSI/Diffu

# Run with very limited parameters for testing
python visual/visual_vae.py \
    --num_examples 1 \
    --device cuda:1 \
    --save_path results/vae_test_visualization \
    --spectral_channels 160 \
    --latent_channels 8 \
    --base_channels 128

echo "VAE visualization test completed."