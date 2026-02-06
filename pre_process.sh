#!/bin/bash
#SBATCH --job-name=data_pipeline
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=224
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=process_%j.out
#SBATCH --error=process_%j.err

echo "--- Data Processing Pipeline Started on $(hostname) ---"
date

source $HOME/miniconda/bin/activate battery_train_env
cd $SLURM_SUBMIT_DIR
set -e

# --- CONFIGURATION FOR MPI STEPS ---
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "----------------------------------------------------------------"
echo "STEP 1: Create Intermediate Dataset (MPI)"
echo "Running on $SLURM_NTASKS tasks across $SLURM_NNODES nodes"
echo "----------------------------------------------------------------"

mpirun --bind-to none python scripts/create_intermediate_dataset.py

echo "----------------------------------------------------------------"
echo "STEP 2: Prune Intermediate Data (MPI)"
echo "----------------------------------------------------------------"

mpirun --bind-to none python scripts/prune_intermediate_data.py

echo "--- Processing Pipeline Complete ---"
date