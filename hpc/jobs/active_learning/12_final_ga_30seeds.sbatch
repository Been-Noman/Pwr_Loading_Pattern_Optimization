#!/bin/bash
#SBATCH --job-name=pwr_ga30
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --array=0-29%30
#SBATCH --output=run/logs/final_ga_%A_%a.out
#SBATCH --error=run/logs/final_ga_%A_%a.err
set -euo pipefail
PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
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
mkdir -p "$PWR_RUN/ga/final_calibrated_ensemble"
srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=none \
  python "$PWR_SCRIPTS/optimization/run_initialization_comparison.py" \
  --round-dir "$PWR_RUN/cnn/final_tail_10000" \
  --audit-json "$PWR_RUN/cnn/final_tail_10000/final_evaluation/final_ensemble_audit.json" \
  --output-dir "$PWR_RUN/ga/final_calibrated_ensemble_warmstart" \
  --warm-start-json "$PWR_RUN/ga/training_warm_start_top200.json" \
  --labelled-json "$PWR_RUN/final/final_tail_10000.json" \
  --seed-index "$SLURM_ARRAY_TASK_ID" \
  --threads "$SLURM_CPUS_PER_TASK"
