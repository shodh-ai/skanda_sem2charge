#!/bin/bash
#SBATCH --job-name=battery_train
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=28
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --exclusive

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Nodes: $SLURM_NNODES ($SLURM_NODELIST)"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total GPUs: 16"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Environment setup
source $HOME/miniconda/bin/activate battery_train_env
cd /sharedstorage/innmivshodhaislurmh1/home/shodhai.admin/skanda_sem2charge
export PYTHONPATH=$PYTHONPATH:$(pwd)

# NCCL settings for InfiniBand (400 Gbps available!)
export NCCL_DEBUG=INFO                       # Change to WARN after first successful run
export NCCL_IB_DISABLE=0                     # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=5                  # Max GPU Direct RDMA
export NCCL_IB_GID_INDEX=3                   # InfiniBand GID
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_8,mlx5_9,mlx5_10,mlx5_11  # Active IB adapters
export NCCL_SOCKET_IFNAME=^docker,lo         # Exclude interfaces

# Performance optimizations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_DEVICE_MAX_CONNECTIONS=1         # Better CUDA kernel concurrency

# Master node setup
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
export MASTER_ADDR=${nodes[0]}
export MASTER_PORT=12355

echo ""
echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "Python: $(which python)"
echo "Environment: $CONDA_DEFAULT_ENV"
echo ""

# System info
echo "GPU info on master node:"
srun --nodes=1 --ntasks=1 -w ${nodes[0]} nvidia-smi -L
echo ""

# Run training
echo "Starting training..."
python -u train.py --experiment_name "base_model"

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="