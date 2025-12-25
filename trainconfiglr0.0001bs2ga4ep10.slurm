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
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCHDYNAMO_DISABLE=1

deepspeed --num_gpus=4 Finetuning_V4-deepseek.py 0.0001 7192 2 2 4 0.01 10
