#!/bin/bash -l
#SBATCH --job-name=c5q5omc960
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --array=0-959%240
#SBATCH --output=run/logs/c5q5omc960_%A_%a.out
#SBATCH --error=run/logs/c5q5omc960_%A_%a.err

set -euo pipefail

PROJECT="$PWR_PROJECT"

source "$PROJECT/config.sh"

IDX="$SLURM_ARRAY_TASK_ID"
START="$IDX"
N=1

SOURCE="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_calibration/protocol/cycle5_q5_calibration_winner_manifest_960_v1.json"

SOURCE_AUDIT="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_calibration/protocol/cycle5_q5_calibration_winner_selection_audit_960_v1.json"

PATTERNS="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_calibration/protocol/cycle5_q5_calibration_openmc_input_960_v1.json"

EXECUTION="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_calibration/protocol/cycle5_q5_calibration_openmc960_execution_manifest_v1.json"

RUN="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_calibration/openmc960"

OUT="$RUN/chunks/chunk_$(printf '%03d' "$IDX").jsonl"

METRICS="$RUN/metrics/chunk_$(printf '%03d' "$IDX")_metrics.json"

TMPROOT="$RUN/tmp/job_${SLURM_ARRAY_JOB_ID}_${IDX}"

PIPELINE="$PWR_PIPELINE_DIR/sa_sqa_openmc_pipeline.py"

EXPECTED_SOURCE_SHA="f204bf469f419658320bc2ef6b8acf5525d9ce5254cbf7bca844516fb1fa234f"
EXPECTED_SOURCE_AUDIT_SHA="87f4e011ae2a278c1def0616529328a1f652e92d5b3db46c61d65b078b79a52a"
EXPECTED_PATTERN_SHA="b4ff8b657038491f3ad4988ecd4bcfd2858c7311207e8f9344618dc87a831a30"
EXPECTED_EXECUTION_SHA="b627adb2094d6c2f2dd02fd57386e2b1e629f34b4badbfbe81c46aa2c60f75cc"
EXPECTED_RUNNER_SHA="fbab7de92c7f7bd7836da6b7cb0c5d77f1d2b77a1ccdae02493c2bbfc5d116fb"
EXPECTED_PIPELINE_SHA="8b9cb9d5fe33612a0b728a2371ce0c098310292826a209131dcb0b266f91ffb3"

mkdir -p     "$RUN/chunks"     "$RUN/metrics"     "$RUN/tmp"     "$PWR_RUN/logs"

ACTUAL_SOURCE_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
ACTUAL_SOURCE_AUDIT_SHA="$(sha256sum "$SOURCE_AUDIT" | awk '{print $1}')"
ACTUAL_PATTERN_SHA="$(sha256sum "$PATTERNS" | awk '{print $1}')"
ACTUAL_EXECUTION_SHA="$(sha256sum "$EXECUTION" | awk '{print $1}')"
ACTUAL_RUNNER_SHA="$(sha256sum "$PWR_PROJECT/code/physics/verify_optimized_patterns.py" | awk '{print $1}')"
ACTUAL_PIPELINE_SHA="$(sha256sum "$PIPELINE" | awk '{print $1}')"

[[ "$ACTUAL_SOURCE_SHA" == "$EXPECTED_SOURCE_SHA" ]] || {
    echo "ERROR: Cycle5 q5 winner manifest changed" >&2
    exit 2
}

[[ "$ACTUAL_SOURCE_AUDIT_SHA" == "$EXPECTED_SOURCE_AUDIT_SHA" ]] || {
    echo "ERROR: Cycle5 q5 winner audit changed" >&2
    exit 3
}

[[ "$ACTUAL_PATTERN_SHA" == "$EXPECTED_PATTERN_SHA" ]] || {
    echo "ERROR: frozen OpenMC960 input changed" >&2
    exit 4
}

[[ "$ACTUAL_EXECUTION_SHA" == "$EXPECTED_EXECUTION_SHA" ]] || {
    echo "ERROR: OpenMC960 execution manifest changed" >&2
    exit 5
}

[[ "$ACTUAL_RUNNER_SHA" == "$EXPECTED_RUNNER_SHA" ]] || {
    echo "ERROR: frozen OpenMC runner changed" >&2
    exit 6
}

[[ "$ACTUAL_PIPELINE_SHA" == "$EXPECTED_PIPELINE_SHA" ]] || {
    echo "ERROR: frozen OpenMC physics pipeline changed" >&2
    exit 7
}

cd "$PWR_PROJECT"

module purge || true
module load openmc/0.15.3-foss-2023a

source "$PWR_OPENMC_VENV/bin/activate"

export OPENMC_CROSS_SECTIONS

export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"

unset     OMP_PROC_BIND     OMP_PLACES     GOMP_CPU_AFFINITY

echo "================================================================"
echo "CYCLE5 q5 CALIBRATION OPENMC960"
echo "ARRAY_TASK=$IDX"
echo "CASE_ID=$IDX"
echo "NODE=$HOSTNAME"
echo "WINNER_SHA=$ACTUAL_SOURCE_SHA"
echo "OPENMC_INPUT_SHA=$ACTUAL_PATTERN_SHA"
echo "EXECUTION_SHA=$ACTUAL_EXECUTION_SHA"
echo "RUNNER_SHA=$ACTUAL_RUNNER_SHA"
echo "PIPELINE_SHA=$ACTUAL_PIPELINE_SHA"
echo "BATCHES=500"
echo "INACTIVE=50"
echo "PARTICLES=500000"
echo "WORKERS=1"
echo "THREADS_PER_RUN=48"
echo "WALLTIME=12:00:00"
echo "Q5_FITTED=NO"
echo "================================================================"

srun     -n 1     -c "$SLURM_CPUS_PER_TASK"     --cpu-bind=none     python "$PWR_PROJECT/code/physics/verify_optimized_patterns.py"     --pattern-file "$PATTERNS"     --output-file "$OUT"     --metrics-file "$METRICS"     --start-index "$START"     --n-samples "$N"     --workers 1     --threads-per-run 48     --batches 500     --inactive 50     --particles 500000     --tmp-root "$TMPROOT"

echo "CYCLE5_Q5_OPENMC960_TASK_${IDX}_DONE"
