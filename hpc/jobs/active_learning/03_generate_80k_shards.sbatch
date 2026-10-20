#!/bin/bash
#SBATCH --job-name=sa_sqa_80k
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --array=0-99%100
#SBATCH --output=run/logs/pattern80k_%A_%a.out
#SBATCH --error=run/logs/pattern80k_%A_%a.err
set -euo pipefail
PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
SHARD="$SLURM_ARRAY_TASK_ID"
SHARD_DIR="$PWR_SHARD_ROOT/shard_${SHARD}"
mkdir -p "$SHARD_DIR"
cd "$PWR_PROJECT"
module purge || true
module load openmc/0.15.3-foss-2023a
source "$PWR_OPENMC_VENV/bin/activate"
export OPENMC_CROSS_SECTIONS
export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"
unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY
srun -n 1 -c 48 --cpu-bind=none python "$PWR_SCRIPTS/annealing/sample_loading_patterns.py" \
  --shard-id "$SHARD" \
  --output-dir "$SHARD_DIR" \
  --exclude-pattern-file "$PWR_EXCLUSION_7500" \
  --expected-excluded 7500 \
  --samples-per-group 200 \
  --workers 48 \
  --max-batches-per-group 5000 \
  --base-seed 30360814 \
  --resume
