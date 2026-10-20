#!/bin/bash -l
#SBATCH --job-name=c5omc3600
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --array=0-143%100
#SBATCH --output=run/logs/c5omc3600_%A_%a.out
#SBATCH --error=run/logs/c5omc3600_%A_%a.err

set -euo pipefail

PROJECT="$PWR_PROJECT"

source "$PROJECT/config.sh"

IDX="$SLURM_ARRAY_TASK_ID"

START=$((IDX * 25))
N=25

PATTERNS="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/protocol/cycle5_acquisition_selected_3600_v1.json"

RUN="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/acquisition/openmc3600"

OUT="$RUN/chunks/chunk_$(printf '%03d' "$IDX").jsonl"

METRICS="$RUN/metrics/chunk_$(printf '%03d' "$IDX")_metrics.json"

TMPROOT="$RUN/tmp/job_${SLURM_ARRAY_JOB_ID}_${IDX}"

EXPECTED_SELECTED_SHA="6487f69a8ace9ca4a10bd8ba8cf06a38059665751c1dd5113bf0b361c6ef4218"
EXPECTED_RUNNER_SHA="fbab7de92c7f7bd7836da6b7cb0c5d77f1d2b77a1ccdae02493c2bbfc5d116fb"
EXPECTED_PIPELINE_SHA="8b9cb9d5fe33612a0b728a2371ce0c098310292826a209131dcb0b266f91ffb3"

mkdir -p \
  "$RUN/chunks" \
  "$RUN/metrics" \
  "$RUN/tmp" \
  "$PWR_RUN/logs"

[[ "$(sha256sum "$PATTERNS" | awk '{print $1}')" == "$EXPECTED_SELECTED_SHA" ]]
[[ "$(sha256sum "$PWR_PROJECT/code/physics/verify_optimized_patterns.py" | awk '{print $1}')" == "$EXPECTED_RUNNER_SHA" ]]
[[ "$(sha256sum "$PWR_PIPELINE_DIR/sa_sqa_openmc_pipeline.py" | awk '{print $1}')" == "$EXPECTED_PIPELINE_SHA" ]]

cd "$PWR_PROJECT"

module purge || true

module load openmc/0.15.3-foss-2023a

source "$PWR_OPENMC_VENV/bin/activate"

export OPENMC_CROSS_SECTIONS

export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"

unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY

echo "============================================================"
echo "CYCLE5 BALANCED MULTI-ARM OPENMC3600"
echo "ARRAY_TASK=$IDX"
echo "START_INDEX=$START"
echo "N_SAMPLES=$N"
echo "ID_RANGE=$START-$((START + N - 1))"
echo "NODE=$HOSTNAME"
echo "BATCHES=500"
echo "INACTIVE=50"
echo "PARTICLES=500000"
echo "WORKERS=5"
echo "============================================================"

srun \
  -n 1 \
  -c "$SLURM_CPUS_PER_TASK" \
  --cpu-bind=none \
  python "$PWR_PROJECT/code/physics/verify_optimized_patterns.py" \
  --pattern-file "$PATTERNS" \
  --output-file "$OUT" \
  --metrics-file "$METRICS" \
  --start-index "$START" \
  --n-samples "$N" \
  --workers 5 \
  --threads-per-run 9 \
  --batches 500 \
  --inactive 50 \
  --particles 500000 \
  --tmp-root "$TMPROOT"

echo "CYCLE5_OPENMC3600_TASK_${IDX}_DONE"
