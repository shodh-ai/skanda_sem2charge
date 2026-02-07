#!/bin/bash
#SBATCH --job-name=battery_train
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=5-24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err


source $HOME/miniconda/bin/activate battery_train_env
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p logs

echo "Job: $SLURM_JOB_ID on $(hostname)"

python -u train.py --experiment_name "base_model"

echo "Finished: $(date)"
