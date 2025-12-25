#!/bin/bash
#SBATCH --job-name=v4_train
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --nodelist=h100-8s-05
#SBATCH --cpus-per-task=16        # A reasonable number of CPUs per task (64/4=16)
#SBATCH --mem=1024G
#SBATCH --time=12-00:00:00
#SBATCH --output=train_v4_%j.out
#SBATCH --error=train_v4_%j.err

# Load the environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate arsh_env

# Change directory
cd ~/my_model

# Set environment variables
export CUDA_HOME=~/anaconda3/envs/arsh_env
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Execute the actual training command
# Note: You should specify which script 'run_train.sh' executes.
# If 'run_train.sh' uses 'deepspeed' directly, you don't need 'launch_deepspeed.py'.
# If you intend to use 'launch_deepspeed.py' or 'patch_deepspeed.py', the line should change.
bash run_train.sh
cp -r /scratch/axs7716Arsh/* ~/my_model/results/ 2>/dev/null || true

