#!/bin/bash
#SBATCH --job-name=openmc-pilot5
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
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
mkdir -p "$RUN/pilot" "$RUN/tmp"

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
OUT="$RUN/pilot/pilot_5.jsonl"
METRICS="$RUN/pilot/pilot_5_metrics.json"
TMPROOT="$RUN/tmp/pilot_${SLURM_JOB_ID}"
test -r "$PATTERNS"

srun -n 1 -c 48 --cpu-bind=none python "$SRC/code/physics/simulate_patterns.py" \
  --pattern-file "$PATTERNS" \
  --output-file "$OUT" \
  --metrics-file "$METRICS" \
  --start-index 0 \
  --n-samples 5 \
  --workers 5 \
  --threads-per-run 9 \
  --batches 500 \
  --inactive 50 \
  --particles 500000 \
  --tmp-root "$TMPROOT"

srun -n 1 -c 48 --cpu-bind=none python "$SRC/code/physics/inspect_pilot_results.py" "$METRICS"
echo "Full-physics 5-sample pilot completed at $(date --iso-8601=seconds)"
