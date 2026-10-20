#!/bin/bash
#SBATCH --job-name=openmc2500
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --array=0-99%100
#SBATCH --output=external_openmc_%x_%j.out
#SBATCH --error=external_openmc_%x_%j.err
set -euo pipefail

SCRATCH_ROOT="${SCRATCH:-$HOME}"
BASE="$SCRATCH_ROOT/openmc2500"
SRC="${PWR_PROJECT:?Set PWR_PROJECT to the repository root}"
VENV="$SCRATCH_ROOT/venvs/openmc-sa-sqa"
XS="$SCRATCH_ROOT/openmc_data/endfb-vii.1-hdf5/cross_sections.xml"
RUN="$BASE/outputs/production_sa_sqa_2500"
PATTERNS="$RUN/patterns_sa_sqa_2500.json"
mkdir -p "$RUN/chunks" "$RUN/metrics" "$RUN/tmp"

module purge
module load openmc/0.15.3-foss-2023a
source "$VENV/bin/activate"
unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY
export OPENMC_CROSS_SECTIONS="$XS"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$SRC"
CHUNK=${SLURM_ARRAY_TASK_ID}
START=$((CHUNK * 25))
OUT="$RUN/chunks/chunk_${CHUNK}.jsonl"
METRICS="$RUN/metrics/chunk_${CHUNK}_metrics.json"
TMPROOT="$RUN/tmp/job_${SLURM_ARRAY_JOB_ID}_${CHUNK}"
test -r "$PATTERNS"

srun -n 1 -c 48 --cpu-bind=none python "$SRC/code/physics/simulate_patterns.py" \
  --pattern-file "$PATTERNS" \
  --output-file "$OUT" \
  --metrics-file "$METRICS" \
  --start-index "$START" \
  --n-samples 25 \
  --workers 5 \
  --threads-per-run 9 \
  --batches 500 \
  --inactive 50 \
  --particles 500000 \
  --tmp-root "$TMPROOT"

echo "Array task $CHUNK completed at $(date --iso-8601=seconds)"
