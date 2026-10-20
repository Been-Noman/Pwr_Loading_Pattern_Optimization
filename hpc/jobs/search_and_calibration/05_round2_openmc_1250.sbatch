#!/bin/bash
#SBATCH --job-name=v2_r1_omc
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --array=0-124%125
#SBATCH --output=run/logs/v2_r2_openmc_%A_%a.out
#SBATCH --error=run/logs/v2_r2_openmc_%A_%a.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

PATTERNS="$PWR_RUN/optimizer_tail_v2/al_rounds/round_2/round2_openmc_1250.json"
RUN="$PWR_RUN/optimizer_tail_v2/al_rounds/round_2/openmc_10pernode"

IDX="$SLURM_ARRAY_TASK_ID"
START=$((IDX * 10))

OUT="$RUN/chunks/chunk_${IDX}.jsonl"
METRICS="$RUN/metrics/chunk_${IDX}_metrics.json"
TMPROOT="$RUN/tmp/job_${SLURM_ARRAY_JOB_ID}_${IDX}"

mkdir -p "$RUN/chunks" "$RUN/metrics" "$RUN/tmp" "$PWR_RUN/logs"

cd "$PWR_PROJECT"

module purge || true
module load openmc/0.15.3-foss-2023a
source "$PWR_OPENMC_VENV/bin/activate"

export OPENMC_CROSS_SECTIONS
export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"

unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=none \
  python "$PWR_SCRIPTS/physics/verify_optimized_patterns.py" \
  --pattern-file "$PATTERNS" \
  --output-file "$OUT" \
  --metrics-file "$METRICS" \
  --start-index "$START" \
  --n-samples 10 \
  --workers 10 \
  --threads-per-run 4 \
  --batches 500 \
  --inactive 50 \
  --particles 500000 \
  --tmp-root "$TMPROOT"
