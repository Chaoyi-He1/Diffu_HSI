#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=sensor_sr_diff
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH -o sensor_sr_diff_%j.log
#SBATCH -e sensor_sr_diff_%j.err

cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu/
source /mnt/shared-scratch/Katehi_L/chaoyi_he/conda/miniconda/bin/activate
conda activate /mnt/shared-scratch/Katehi_L/chaoyi_he/conda_envs/hsi

export PYTORCH_ALLOC_CONF=expandable_segments:True

which python
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"

GPUS_PER_NODE=2
MASTER_PORT=29500

# Sensor stats file must exist before launch:
#   python -m scripts.compute_sensor_stats --R-n 1
STATS_PATH="dataset/HFD100 Mat dataset/sensor_stats_R1.json"

torchrun \
    --standalone \
    --nproc_per_node=$GPUS_PER_NODE \
    main_2d_sensor_sr_fsdp.py \
    --sensor_stats_path "$STATS_PATH" \
    --batch_size 16 \
    --lr 1e-4 \
    --base_channels 256 \
    --use_channel_3d_conv \
    --channel_num_filters 4 \
    --prediction_type v \
    --lrf 0.033 \
    --ds_sr 8 \
    --sr_downsample_method strided \
    --save_path results/2d_sensor_sr/ds8_strided/HFD/R_1/l1_loss/3dconv \
    --num_epochs 200 \
    --resume results/2d_sensor_sr/ds8_strided/HFD/R_1/l1_loss/3dconv/checkpoint_epoch_100.pth
