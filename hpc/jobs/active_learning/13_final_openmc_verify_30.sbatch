#!/bin/bash
#SBATCH --job-name=ga_verify30
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --array=0-29%30
#SBATCH --output=run/logs/final_verify30_%A_%a.out
#SBATCH --error=run/logs/final_verify30_%A_%a.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

PATTERNS="$PWR_RUN/ga/final_openmc_verify_30.json"
RUN="$PWR_RUN/ga/final_openmc_verification_30"

IDX="$SLURM_ARRAY_TASK_ID"

OUT="$RUN/chunks/chunk_${IDX}.jsonl"
METRICS="$RUN/metrics/chunk_${IDX}_metrics.json"
TMPROOT="$RUN/tmp/job_${SLURM_ARRAY_JOB_ID}_${IDX}"

mkdir -p "$RUN/chunks" "$RUN/metrics" "$RUN/tmp"

cd "$PWR_PROJECT"

module purge || true
module load openmc/0.15.3-foss-2023a
source "$PWR_OPENMC_VENV/bin/activate"

export OPENMC_CROSS_SECTIONS
export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"

unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY

srun -n 1 -c 48 --cpu-bind=none \
  python "$PWR_SCRIPTS/physics/verify_optimized_patterns.py" \
  --pattern-file "$PATTERNS" \
  --output-file "$OUT" \
  --metrics-file "$METRICS" \
  --start-index "$IDX" \
  --n-samples 1 \
  --workers 1 \
  --threads-per-run 48 \
  --batches 500 \
  --inactive 50 \
  --particles 500000 \
  --tmp-root "$TMPROOT"
