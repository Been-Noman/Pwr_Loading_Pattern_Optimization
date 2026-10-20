#!/bin/bash
#SBATCH --job-name=v3postesc
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=run/logs/v3postesc_%j.out
#SBATCH --error=run/logs/v3postesc_%j.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

OUT="$PWR_RUN/optimizer_tail_v2/robust_ga_v3_smoke/post_escape_descent_seed2026"

mkdir -p "$OUT" "$PWR_RUN/logs"

cd "$PWR_PROJECT"

module purge || true
module load python/3.11.5-gcccore-13.2.0
source "$PWR_ML_VENV/bin/activate"

export PYTHONPATH="$PWR_SCRIPTS/optimization:$PWR_SCRIPTS:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTEROP_THREADS=4
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=none \
  python "$PWR_PROJECT/code/optimization/examine_local_descent.py" \
  --round-dir "$PWR_RUN/optimizer_tail_v2/models/round2_cnn" \
  --residual-model-dir "$PWR_RUN/optimizer_tail_v2/models/learned_residual_safety" \
  --near-archive-json "$PWR_RUN/optimizer_tail_v2/robust_ga_v3_smoke/local_escape_seed2026/best_escape_seed_archive.json" \
  --labelled-json "$PWR_RUN/optimizer_tail_v2/manifests/known_openmc_all_d4_exclusion.json" \
  --output-dir "$OUT" \
  --batch-size 2048 \
  --threads "$SLURM_CPUS_PER_TASK" \
  --max-steps 100 \
  --improvement-tol 1e-12
