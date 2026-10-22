#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=ssr_ds_sweep
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH -o sensor_sr_sweep_%j.log
#SBATCH -e sensor_sr_sweep_%j.err

# DS_SR is required and must be passed via:
#   sbatch --export=ALL,DS_SR=<rate> job_sensor_sr_sweep.slurm
if [[ -z "$DS_SR" ]]; then
    echo "ERROR: DS_SR environment variable not set." >&2
    echo "Submit with: sbatch --export=ALL,DS_SR=<rate> job_sensor_sr_sweep.slurm" >&2
    exit 1
fi

cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu/
source /mnt/shared-scratch/Katehi_L/chaoyi_he/conda/miniconda/bin/activate
conda activate /mnt/shared-scratch/Katehi_L/chaoyi_he/conda_envs/hsi

export PYTORCH_ALLOC_CONF=expandable_segments:True

which python
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"

GPUS_PER_NODE=2

STATS_PATH="dataset/HFD100 Mat dataset/sensor_stats_R1.json"
SAVE_PATH="results/2d_sensor_sr/ds${DS_SR}_area/HFD/R_1/l1_loss/3dconv"

echo "=========================================="
echo "Sensor-SR sweep run: DS_SR=${DS_SR}"
echo "Save path: ${SAVE_PATH}"
echo "=========================================="

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
    --ds_sr "$DS_SR" \
    --sr_downsample_method area_resize \
    --save_path "$SAVE_PATH" \
    --num_epochs 200
