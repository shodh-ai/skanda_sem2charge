#!/bin/bash
#SBATCH --job-name=battery_train
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --gres=gpu:H100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
echo "GPUs per node: 8"
echo "Total GPUs: 16"
echo "=========================================="

# ---------------------------------------------------------------------
# 1. Environment Setup
# ---------------------------------------------------------------------
source $HOME/miniconda/bin/activate battery_train_env
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set environment variables for optimal multi-node performance
export NCCL_DEBUG=INFO                    # Debug NCCL communication
export NCCL_IB_DISABLE=0                  # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=2               # Enable GPU Direct RDMA
export NCCL_SOCKET_IFNAME=^docker,lo      # Exclude docker and loopback interfaces

# OMP settings for CPU efficiency
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------------------------------------------------------------------
# 2. System Info
# ---------------------------------------------------------------------
echo ""
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "PyTorch Lightning version: $(python -c 'import pytorch_lightning; print(pytorch_lightning.__version__)')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "NCCL version: $(python -c 'import torch; print(torch.cuda.nccl.version())')"
echo ""

# ---------------------------------------------------------------------
# 3. Run Training with srun (SLURM-managed multi-node)
# ---------------------------------------------------------------------
echo "Starting training on $SLURM_NNODES nodes with 8 GPUs each..."
echo ""

srun python -u train.py --experiment_name "base_model"

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="