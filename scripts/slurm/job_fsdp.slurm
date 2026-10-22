#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=2d_diffusion        	   #Set the job name to "JobExample4"
#SBATCH --time=96:00:00              	   #Set the wall clock limit to 1hr and 30min
#SBATCH --nodes=1
#SBATCH --ntasks=1                   	   #Request 1 task (CPU)
#SBATCH --cpus-per-task=16            	   #Request exactly (or max) of 28 tasks per node     
#SBATCH --gres=gpu:a100:2            	   #Request 1 GPU per node can be 1 or 2, type=k80 or v100
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH -o 2d_diffusion_ds2_%j.log         #Send stdout/err to "Example4Out.[jobID]"
#SBATCH -e 2d_diffusion_ds2_%j.err
##SBATCH --mail-type=FAIL                  #Send email on all job events
##SBATCH --mail-user=chaoyi_he@tamu.edu    #Send all emails to email_address 

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=122768702865            #Set billing account to Terra:122768702865	Grace:132768702708
## --master_port=42247

#First Executable Line

cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu/
source /mnt/shared-scratch/Katehi_L/chaoyi_he/conda/miniconda/bin/activate
conda activate /mnt/shared-scratch/Katehi_L/chaoyi_he/conda_envs/hsi

# Reduce allocator fragmentation after OOM retries (see PyTorch CUDA memory notes)
export PYTORCH_ALLOC_CONF=expandable_segments:True

which python
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"

# -----------------------------------------------------------------------
# Distributed settings
# torchrun reads MASTER_ADDR / MASTER_PORT from env if set via --master_addr/port.
# For single-node, --standalone handles everything automatically.
# For multi-node, remove --standalone and set the vars below.
# -----------------------------------------------------------------------
GPUS_PER_NODE=2
MASTER_PORT=29500

# -----------------------------------------------------------------------
# Single-node launch (most common on TAMU GRACE/TERRA)
# -----------------------------------------------------------------------
# Global batch is split across GPUs; full-res attention in stage6 is VRAM-heavy.
# If OOM persists: lower --batch_size further and/or reduce --base_channels.
#
# NOTE on new defaults (set in main_2d_fsdp.py; overridden below where useful):
#   --prediction_type v       v-prediction is much more stable than eps at both
#                             high and low SNR (default in main_2d_fsdp.py)
#   --lrf 0.033               eta_min ≈ lr*0.033 = 1e-6 at lr=3e-5 —
#                             avoids bf16 precision collapse at tail of cosine
#   --channel_num_filters 4   4 parallel 3D filters per ChannelSpatialConv
#                             (was 1; multiplies spectral expressivity ~free)
#   --ds_warmup_epochs 20     First 20 epochs run with sensor_down_sample_rate=1
#                             regardless of --ds; then anneals to --ds
#
# IMPORTANT: pre-April-17 checkpoints are INCOMPATIBLE with these defaults
# (layer shapes + prediction target changed). The --resume path below points
# to the NEW save dir, so it will miss and start fresh on the first run.
torchrun \
    --standalone \
    --nproc_per_node=$GPUS_PER_NODE \
    main_2d_fsdp.py \
    --batch_size 16 \
    --lr 5e-5 \
    --base_channels 512 \
    --use_channel_3d_conv \
    --channel_num_filters 4 \
    --prediction_type v \
    --lrf 0.033 \
    --ds_warmup_epochs 20 \
    --ds 2 \
    --num_epochs 300 \
    --resume results/2d_hsi_diffusion/down_sample_2/down_sample_2/HFD/R_1/l1_loss/checkpoint_epoch_110.pth \
    --save_path results/2d_hsi_diffusion/down_sample_2/HFD/R_1/l1_loss/vpred_cf4_fsdp_3dconv/

# -----------------------------------------------------------------------
# Multi-node launch example (uncomment and adjust for 2-node jobs)
# Requires: --nodes=2, --ntasks-per-node=4 above
# -----------------------------------------------------------------------
# MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR MASTER_PORT
# 
# torchrun \
#     --nproc_per_node=$GPUS_PER_NODE \
#     --nnodes=$SLURM_NNODES \
#     --node_rank=$SLURM_NODEID \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     main_2d_fsdp.py \
#     --batch_size 24 \
#     --base_channels 256 \
#     --num_epochs 1000 \
#     --save_every 10 \
#     --val_every 50 \
#     --grad_clip 0.1 \
#     --resume results/2d_hsi_diffusion/down_sample_2/HFD/R_1/l1_loss/checkpoint_epoch_71.pth \
#     --save_path results/2d_hsi_diffusion/down_sample_2/HFD/R_1/l1_loss_fsdp
