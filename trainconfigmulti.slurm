#!/bin/bash
#SBATCH --job-name=v4_multigpu
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=12:00:00
#SBATCH --output=train_v4_%j.out
#SBATCH --error=train_v4_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate arsh_env
cd ~/my_model
export DS_BUILD_OPS=0
export TORCHDYNAMO_DISABLE=1
RANDOM_PORT=$((29500 + RANDOM % 1000))
torchrun --nproc_per_node=4 --master_port=$RANDOM_PORT Finetuning_V4-deepseek.py 0.0001 7192 2 2 4 0.01 10
