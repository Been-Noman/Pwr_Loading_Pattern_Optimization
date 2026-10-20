#!/bin/bash
#SBATCH --job-name=build_bank50k
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --output=run/logs/build_bank50k_%j.out
#SBATCH --error=run/logs/build_bank50k_%j.err
set -euo pipefail
PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
cd "$PWR_PROJECT"
module purge || true
module load python/3.11.5-gcccore-13.2.0
source "$PWR_ML_VENV/bin/activate"
export PYTHONPATH="$PWR_SCRIPTS:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
N_SHARDS="${N_SHARDS:-100}"
srun -n 1 -c 48 --cpu-bind=none python "$PWR_SCRIPTS/data/build_diverse_candidate_pool.py" \
  --shard-root "$PWR_SHARD_ROOT" \
  --n-shards "$N_SHARDS" \
  --exclude-pattern-file "$PWR_EXCLUSION_7500" \
  --target-total 50000 \
  --target-per-group 12500 \
  --clusters 512 \
  --projection-components 64 \
  --seed 42 \
  --output-file "$PWR_CANDIDATE_BANK" \
  --model-output "$PWR_DIVERSITY_MODEL"
