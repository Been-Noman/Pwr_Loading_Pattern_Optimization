#!/bin/bash
#SBATCH --job-name=v2cf25
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --array=0-24%5
#SBATCH --output=run/logs/v2cf25_%A_%a.out
#SBATCH --error=run/logs/v2cf25_%A_%a.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

SEEDS=(42 314 2718 31415 161803)

TASK="$SLURM_ARRAY_TASK_ID"
FOLD=$(( TASK / 5 ))
SEED_INDEX=$(( TASK % 5 ))
SEED="${SEEDS[$SEED_INDEX]}"

ROUND_DIR="$PWR_RUN/optimizer_tail_v2/crossfit_round2/folds/fold_${FOLD}"

cd "$PWR_PROJECT"

module purge || true
module load python/3.11.5-gcccore-13.2.0
source "$PWR_ML_VENV/bin/activate"

export PYTHONPATH="$PWR_SCRIPTS:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTEROP_THREADS=4
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "TASK=$TASK FOLD=$FOLD SEED=$SEED ROUND_DIR=$ROUND_DIR"

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=none \
  python "$PWR_PROJECT/code/surrogate/train_balanced_cnn_streaming.py" \
  --round-dir "$ROUND_DIR" \
  --seed "$SEED" \
  --threads "$SLURM_CPUS_PER_TASK"
