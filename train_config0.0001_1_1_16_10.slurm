#!/bin/bash
#SBATCH --job-name=lre5bs1ga16ep10
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=train_config2_%j.out
#SBATCH --error=train_config2_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate arsh_env
cd ~/my_model
export TORCHDYNAMO_DISABLE=1
python Finetuning_V3_11-19-2025.py 0.00005 7264 1 1 8 0.01 10

