#!/bin/bash -l
#SBATCH --job-name=q5prosomc360
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --array=0-359%120
#SBATCH --output=run/logs/q5prosomc360_%A_%a.out
#SBATCH --error=run/logs/q5prosomc360_%A_%a.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

IDX="$SLURM_ARRAY_TASK_ID"
START="$IDX"
N=1

SOURCE="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360/protocol/cycle5_q5_prospective360_winner_manifest_360_v1.json"

SOURCE_AUDIT="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360/protocol/cycle5_q5_prospective360_winner_manifest_360_v1_audit.json"

PATTERNS="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360/protocol/cycle5_q5_prospective360_openmc_input_360_v1.json"

EXECUTION="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360/protocol/cycle5_q5_prospective360_openmc360_execution_manifest_v1.json"

RUN="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360/openmc360"

OUT="$RUN/chunks/chunk_$(printf '%03d' "$IDX").jsonl"
METRICS="$RUN/metrics/chunk_$(printf '%03d' "$IDX")_metrics.json"
TMPROOT="$RUN/tmp/job_${SLURM_ARRAY_JOB_ID}_${IDX}"

PIPELINE="$PWR_PIPELINE_DIR/sa_sqa_openmc_pipeline.py"

EXPECTED_SOURCE_SHA="73e4049c0c1c89f0306537a07ece92603ae6b25f6afc3af46f4e7c87c6321ac0"
EXPECTED_SOURCE_AUDIT_SHA="e33a0cddc91368adbe0a975d243ced8bcd9c1437d84b225079867d75f4bf0cc6"
EXPECTED_PATTERN_SHA="a3676f37877f5864fdfded62f5f2c72841f6c476a4659f8fe887ad8ad4f6cae0"
EXPECTED_EXECUTION_SHA="a474e33a7f5e55fce4ca0cde5a42d253bff9393bc9336a2bd2b8dccb30735a6d"
EXPECTED_RUNNER_SHA="fbab7de92c7f7bd7836da6b7cb0c5d77f1d2b77a1ccdae02493c2bbfc5d116fb"
EXPECTED_PIPELINE_SHA="8b9cb9d5fe33612a0b728a2371ce0c098310292826a209131dcb0b266f91ffb3"

mkdir -p \
    "$RUN/chunks" \
    "$RUN/metrics" \
    "$RUN/tmp" \
    "$PWR_RUN/logs"

ACTUAL_SOURCE_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
ACTUAL_SOURCE_AUDIT_SHA="$(sha256sum "$SOURCE_AUDIT" | awk '{print $1}')"
ACTUAL_PATTERN_SHA="$(sha256sum "$PATTERNS" | awk '{print $1}')"
ACTUAL_EXECUTION_SHA="$(sha256sum "$EXECUTION" | awk '{print $1}')"
ACTUAL_RUNNER_SHA="$(sha256sum "$PWR_PROJECT/code/physics/verify_optimized_patterns.py" | awk '{print $1}')"
ACTUAL_PIPELINE_SHA="$(sha256sum "$PIPELINE" | awk '{print $1}')"

[[ "$ACTUAL_SOURCE_SHA" == "$EXPECTED_SOURCE_SHA" ]] || exit 2
[[ "$ACTUAL_SOURCE_AUDIT_SHA" == "$EXPECTED_SOURCE_AUDIT_SHA" ]] || exit 3
[[ "$ACTUAL_PATTERN_SHA" == "$EXPECTED_PATTERN_SHA" ]] || exit 4
[[ "$ACTUAL_EXECUTION_SHA" == "$EXPECTED_EXECUTION_SHA" ]] || exit 5
[[ "$ACTUAL_RUNNER_SHA" == "$EXPECTED_RUNNER_SHA" ]] || exit 6
[[ "$ACTUAL_PIPELINE_SHA" == "$EXPECTED_PIPELINE_SHA" ]] || exit 7

cd "$PWR_PROJECT"

module purge || true
module load openmc/0.15.3-foss-2023a

source "$PWR_OPENMC_VENV/bin/activate"

export OPENMC_CROSS_SECTIONS
export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"

unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY

echo "============================================================"
echo "CYCLE5 q5 PROSPECTIVE360 OPENMC"
echo "CASE_ID=$IDX"
echo "NODE=$HOSTNAME"
echo "BATCHES=500"
echo "INACTIVE=50"
echo "PARTICLES=500000"
echo "WORKERS=1"
echo "THREADS_PER_RUN=48"
echo "DUPLICATES_PRESERVED=YES"
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
        --workers 1 \
        --threads-per-run 48 \
        --batches 500 \
        --inactive 50 \
        --particles 500000 \
        --tmp-root "$TMPROOT"

echo "CYCLE5_Q5_PROSPECTIVE360_OPENMC_TASK_${IDX}_DONE"
