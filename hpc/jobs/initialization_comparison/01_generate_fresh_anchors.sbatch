#!/bin/bash
#SBATCH --job-name=formal_anchors
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --array=0-29%30
#SBATCH --output=run/logs/formal_anchors_%A_%a.out
#SBATCH --error=run/logs/formal_anchors_%A_%a.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

SHARD="$SLURM_ARRAY_TASK_ID"
ROOT="$PWR_RUN/formal_random_sa_sqa/manifests/anchor_shards"
SHARD_DIR="$ROOT/shard_${SHARD}"

mkdir -p "$SHARD_DIR"
cd "$PWR_PROJECT"

module purge || true
module load openmc/0.15.3-foss-2023a
source "$PWR_OPENMC_VENV/bin/activate"

export OPENMC_CROSS_SECTIONS
export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"

unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY

srun -n 1 -c 48 --cpu-bind=none \
python "$PWR_SCRIPTS/annealing/sample_loading_patterns.py" \
  --shard-id "$SHARD" \
  --output-dir "$SHARD_DIR" \
  --exclude-pattern-file "$PWR_RUN/final/final_tail_10000.json" \
  --expected-excluded 10000 \
  --samples-per-group 100 \
  --workers 48 \
  --max-batches-per-group 5000 \
  --base-seed 20260816 \
  --resume
