#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=2d_diffusion_HFD_ds2        	#Set the job name to "JobExample4"
#SBATCH --time=23:30:00               		#Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=8                   		#Request 1 task (CPU)
#SBATCH --ntasks-per-node=8          		#Request exactly (or max) of 28 tasks per node
#SBATCH --mem=64G                    		#Request 64GB per node
#SBATCH --gres=gpu:a100:1                 	#Request 1 GPU per node can be 1 or 2, type=k80 or v100
#SBATCH --partition=gpu              		#Request the GPU partition/queue
#SBATCH --output=2d_diffusion_HFD_ds2_%j       	#Send stdout/err to "Example4Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=122768702865            	#Set billing account to Terra:122768702865	Grace:132768702708
##SBATCH --mail-type=ALL              	   	#Send email on all job events
##SBATCH --mail-user=chaoyi_he@tamu.edu    	#Send all emails to email_address 

#First Executable Line

module load GCCcore/12.2.0
module load Python/3.10.8

cd /scratch/user/chaoyi_he
cd Diffu
source /scratch/user/chaoyi_he/Python_3_10/bin/activate		

which python

python -c "import torch; print('Torch version:', torch.__version__)"
python main_2d.py
