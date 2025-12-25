#!/bin/bash
#SBATCH --job-name=check_cuda
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=cuda_check_%j.out
#SBATCH --error=cuda_check_%j.err

echo "=== CUDA Environment Check ==="
echo "Current node: $(hostname)"
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo ""

echo "=== Searching for CUDA ==="
which nvcc
which nvidia-smi
nvidia-smi
ls /usr/local/ | grep -i cuda
find /usr -name "nvcc" 2>/dev/null | head -5

echo "=== Checking PyTorch CUDA ==="
source ~/anaconda3/etc/profile.d/conda.sh
conda activate arsh_env
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version from PyTorch:', torch.version.cuda)"
