#!/bin/bash
#SBATCH --job-name=v3_train
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=12:00:00
#SBATCH --output=train_v3_%j.out
#SBATCH --error=train_v3_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate arsh_env
cd ~/my_model

export CUDA_HOME=~/anaconda3/envs/arsh_env

# Run V3 with same parameters as before
python Finetuning_V3_11-19-2025.py 0.0002 8870 2 2 8 0.01 80

# Copy results to home directory
cp -r /scratch/axs7716Arsh/* ~/my_model/results/ 2>/dev/null || true
