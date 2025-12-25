  #!/bin/bash
  #SBATCH --job-name=centaur_train
  #SBATCH --partition=defq
  #SBATCH --nodes=1
  #SBATCH --gpus=4
  #SBATCH --cpus-per-task=64
  #SBATCH --mem=1024G
  #SBATCH --time=12:00:00
  #SBATCH --output=train_%j.out
  #SBATCH --error=train_%j.err

  # Activate environment
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate arsh_env

  # Go to project
  cd ~/my_model

  # Set environment variables
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export TORCHDYNAMO_DISABLE=1

  # Run training
  python Finetuning_V3_11-19-2025.py 0.0002 16384 2 4 4 0.01 10

