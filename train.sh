#!/bin/bash
#SBATCH --job-name=battery_train_100ep
#SBATCH --partition=shodhp
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

echo "Job started on $(hostname) at $(date)"

# ---------------------------------------------------------------------
# 1. Environment Setup
# ---------------------------------------------------------------------
source $HOME/miniconda/bin/activate battery_train_env
export PYTHONPATH=$PYTHONPATH:$(pwd)

# ---------------------------------------------------------------------
# 2. System Info & Debugging
# ---------------------------------------------------------------------
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "GPUs Available: $CUDA_VISIBLE_DEVICES"
echo "SLURM Job ID: $SLURM_JOB_ID"

# ---------------------------------------------------------------------
# 3. Run Training
# ---------------------------------------------------------------------
python -u train.py --experiment_name "base_model"

echo "Job finished at $(date)"