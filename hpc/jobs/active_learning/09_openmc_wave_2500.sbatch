#!/bin/bash
#SBATCH --job-name=openmc_wave
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --array=0-99%100
#SBATCH --output=run/logs/openmc_wave_%A_%a.out
#SBATCH --error=run/logs/openmc_wave_%A_%a.err
set -euo pipefail
: "${WAVE:?export WAVE before sbatch}"
PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
RUN="$PWR_RUN/waves/wave_${WAVE}"
PATTERNS="$RUN/selected_2500.json"
CHUNK="$SLURM_ARRAY_TASK_ID"
START=$((CHUNK * 25))
OUT="$RUN/chunks/chunk_${CHUNK}.jsonl"
METRICS="$RUN/metrics/chunk_${CHUNK}_metrics.json"
TMPROOT="$RUN/tmp/job_${SLURM_ARRAY_JOB_ID}_${CHUNK}"
mkdir -p "$RUN/chunks" "$RUN/metrics" "$RUN/tmp"
cd "$PWR_PROJECT"
module purge || true
module load openmc/0.15.3-foss-2023a
source "$PWR_OPENMC_VENV/bin/activate"
export OPENMC_CROSS_SECTIONS
export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"
unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY
srun -n 1 -c 48 --cpu-bind=none python "$PWR_OPENMC_RUNNER" \
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
