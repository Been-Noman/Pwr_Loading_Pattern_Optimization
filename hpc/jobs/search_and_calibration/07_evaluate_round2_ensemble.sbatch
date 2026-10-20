#!/bin/bash
#SBATCH --job-name=v2r2ensal
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=run/logs/v2r2ensal_%j.out
#SBATCH --error=run/logs/v2r2ensal_%j.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

ROUND_DIR="$PWR_RUN/optimizer_tail_v2/models/round2_cnn"
OUTPUT_DIR="$ROUND_DIR/ensemble_evaluation"

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

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=none \
  python "$PWR_PROJECT/code/surrogate/evaluate_second_refinement.py" \
  --round-dir "$ROUND_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --threads "$SLURM_CPUS_PER_TASK"
