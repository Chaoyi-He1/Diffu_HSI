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
torchrun \
    --standalone \
    --nproc_per_node=$GPUS_PER_NODE \
    main_2d_fsdp.py \
    --batch_size 12 \

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
