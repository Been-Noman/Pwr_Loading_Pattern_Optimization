#!/bin/bash
#SBATCH --job-name=v3local
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=run/logs/v3local_%j.out
#SBATCH --error=run/logs/v3local_%j.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

OUT="$PWR_RUN/optimizer_tail_v2/robust_ga_v3_smoke/local_probe_seed2026"

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
  python "$PWR_PROJECT/code/optimization/examine_swap_neighborhood.py" \
  --round-dir "$PWR_RUN/optimizer_tail_v2/models/round2_cnn" \
  --residual-model-dir "$PWR_RUN/optimizer_tail_v2/models/learned_residual_safety" \
  --near-archive-json "$PWR_RUN/optimizer_tail_v2/robust_ga_v3_smoke/random/ga_seed_2026_v3_near_archive.json" \
  --labelled-json "$PWR_RUN/optimizer_tail_v2/manifests/known_openmc_all_d4_exclusion.json" \
  --output-dir "$OUT" \
  --sample-2 20000 \
  --sample-4 20000 \
  --batch-size 2048 \
  --seed 93017 \
  --threads "$SLURM_CPUS_PER_TASK"
