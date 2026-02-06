#!/bin/bash
#SBATCH --job-name=optimise
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=optimise_%j.out
#SBATCH --error=optimise_%j.err

echo "--- Optimise Script Started on $(hostname) ---"
date

# 1. Activate Environment
source $HOME/miniconda/bin/activate battery_train_env

# 2. Go to project root
cd $SLURM_SUBMIT_DIR

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting Python Script..."

# 5. Run the script
# Ensure your config.yaml has num_workers: 220 (or -1/auto) to use the full node.
python -u scripts/optimise_from_parquet.py

echo "--- Optimization Complete ---"
date