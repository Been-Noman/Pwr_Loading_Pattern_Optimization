#!/bin/bash
#SBATCH --job-name=omcval18
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --array=0-17%18
#SBATCH --output=run/logs/omcval18_%A_%a.out
#SBATCH --error=run/logs/omcval18_%A_%a.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

CASE_ID="$SLURM_ARRAY_TASK_ID"

VALDIR="$PWR_RUN/optimizer_tail_v2/openmc_validation18"
PATTERNS="$VALDIR/validation18_patterns.json"

RESULTS_DIR="$VALDIR/results"
TMP_BASE="$VALDIR/tmp"

mkdir -p "$RESULTS_DIR" "$TMP_BASE" "$PWR_RUN/logs"

OUTPUT_FILE="$RESULTS_DIR/case_$(printf '%02d' "$CASE_ID")_output.jsonl"
METRICS_FILE="$RESULTS_DIR/case_$(printf '%02d' "$CASE_ID")_metrics.json"
TMP_ROOT="$TMP_BASE/case_$(printf '%02d' "$CASE_ID")"

rm -f "$OUTPUT_FILE" "$METRICS_FILE"
rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"

cd "$PWR_PROJECT"

module purge || true
module load openmc/0.15.3-foss-2023a

source "$PWR_OPENMC_VENV/bin/activate"

export OPENMC_CROSS_SECTIONS="$OPENMC_CROSS_SECTIONS"
export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"

unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY

echo "============================================================"
echo "PROSPECTIVE OPENMC VALIDATION18"
echo "JOB_ID=$SLURM_JOB_ID"
echo "ARRAY_TASK=$CASE_ID"
echo "NODE=$(hostname)"
echo "PATTERN_FILE=$PATTERNS"
echo "START_INDEX=$CASE_ID"
echo "BATCHES=500"
echo "INACTIVE=50"
echo "PARTICLES=500000"
echo "CPUS=$SLURM_CPUS_PER_TASK"
echo "OUTPUT_FILE=$OUTPUT_FILE"
echo "METRICS_FILE=$METRICS_FILE"
echo "TMP_ROOT=$TMP_ROOT"
echo "============================================================"

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=none \
    python "$PWR_SCRIPTS/physics/verify_optimized_patterns.py" \
    --pattern-file "$PATTERNS" \
    --output-file "$OUTPUT_FILE" \
    --metrics-file "$METRICS_FILE" \
    --start-index "$CASE_ID" \
    --n-samples 1 \
    --workers 1 \
    --threads-per-run "$SLURM_CPUS_PER_TASK" \
    --batches 500 \
    --inactive 50 \
    --particles 500000 \
    --tmp-root "$TMP_ROOT"

echo "OPENMC_VALIDATION_CASE_${CASE_ID}_DONE"
