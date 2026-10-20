#!/bin/bash
#SBATCH --job-name=prep_cnn_round
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=run/logs/prep_cnn_%j.out
#SBATCH --error=run/logs/prep_cnn_%j.err
set -euo pipefail
: "${DATASET_PATH:?export DATASET_PATH before sbatch}"
: "${ROUND_DIR:?export ROUND_DIR before sbatch}"
: "${ROUND_NAME:?export ROUND_NAME before sbatch}"
PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
mkdir -p "$ROUND_DIR"
cd "$PWR_PROJECT"
module purge || true
module load python/3.11.5-gcccore-13.2.0
source "$PWR_ML_VENV/bin/activate"
export PYTHONPATH="$PWR_SCRIPTS:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python "$PWR_SCRIPTS/data/prepare_training_data.py" --dataset "$DATASET_PATH" --output-dir "$ROUND_DIR" --round-name "$ROUND_NAME"
