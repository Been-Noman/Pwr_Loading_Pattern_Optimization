#!/bin/bash
#SBATCH --job-name=score50k
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=run/logs/score50k_%j.out
#SBATCH --error=run/logs/score50k_%j.err
set -euo pipefail
: "${ROUND_DIR:?export ROUND_DIR before sbatch}"
: "${SCORE_DIR:?export SCORE_DIR before sbatch}"
PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
mkdir -p "$SCORE_DIR"
cd "$PWR_PROJECT"
module purge || true
module load python/3.11.5-gcccore-13.2.0
source "$PWR_ML_VENV/bin/activate"
export PYTHONPATH="$PWR_SCRIPTS:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTEROP_THREADS=4
srun -n 1 -c 48 --cpu-bind=none python "$PWR_SCRIPTS/data/score_candidate_pool.py" \
  --round-dir "$ROUND_DIR" --candidate-bank "$PWR_CANDIDATE_BANK" --output-dir "$SCORE_DIR" --threads 48
