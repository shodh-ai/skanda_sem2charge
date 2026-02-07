#!/bin/bash
#SBATCH --job-name=battery_test
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

echo "=========================================="
echo "Testing Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# Create logs directory
mkdir -p logs test_results

# Environment setup
source $HOME/miniconda/bin/activate battery_train_env
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Performance settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# System info
echo ""
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "GPU: $(nvidia-smi -L)"
echo ""

CHECKPOINT_PATH="output/version_1/checkpoints/epochepoch=03-val_lossval_loss=0.1118.ckpt"
TEST_NAME="test_production_780k"

# ============================================================
# Run Testing
# ============================================================
echo "Testing checkpoint: $CHECKPOINT_PATH"
echo ""

python -u test.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --test_name "$TEST_NAME"

echo ""
echo "=========================================="
echo "Testing Completed: $(date)"
echo "=========================================="
