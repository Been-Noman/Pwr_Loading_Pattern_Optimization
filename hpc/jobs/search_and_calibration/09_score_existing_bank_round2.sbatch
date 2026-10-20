#!/bin/bash
#SBATCH --job-name=v2r2bank
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=run/logs/v2r2bank_%j.out
#SBATCH --error=run/logs/v2r2bank_%j.err

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

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=none \
  python "$PWR_PROJECT/code/data/score_optimizer_bank.py" \
  --round-dir "$PWR_RUN/optimizer_tail_v2/models/round2_cnn" \
  --candidate-bank "$PWR_RUN/optimizer_tail_v2/candidate_bank/optimizer_tail_bank_d4_unique_excluded17500.npz" \
  --output-dir "$PWR_RUN/optimizer_tail_v2/analysis/existing_bank_round2_current" \
  --threads "$SLURM_CPUS_PER_TASK" \
  --chunk-size 2048
