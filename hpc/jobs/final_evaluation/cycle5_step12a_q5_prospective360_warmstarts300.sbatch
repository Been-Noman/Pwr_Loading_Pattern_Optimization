#!/bin/bash
#SBATCH --job-name=c5q5prosws
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --array=0-299%100
#SBATCH --output=run/logs/c5q5prosws_%A_%a.out
#SBATCH --error=run/logs/c5q5prosws_%A_%a.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"

source "$PROJECT/config.sh"

cd "$PROJECT"

source \
  run/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360/protocol/step12a.env


SHARD="$SLURM_ARRAY_TASK_ID"

OUTROOT="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360/warmstarts/fresh300/raw"

SHARD_OUT="$OUTROOT/shard_${SHARD}"

mkdir -p \
  "$SHARD_OUT" \
  "$PWR_RUN/logs"


# ----------------------------------------------------------------------
# Freeze-check Step12A master protocol before doing anything.
# ----------------------------------------------------------------------

ACTUAL_MASTER_SHA="$(
    sha256sum "$CYCLE5_Q5_PROSPECTIVE_MASTER" |
    awk '{print $1}'
)"

if [[ \
    "$ACTUAL_MASTER_SHA" != \
    "$CYCLE5_Q5_PROSPECTIVE_MASTER_SHA" \
]]; then
    echo "ERROR: prospective360 protocol SHA mismatch"
    echo "expected = $CYCLE5_Q5_PROSPECTIVE_MASTER_SHA"
    echo "actual   = $ACTUAL_MASTER_SHA"
    exit 2
fi


module purge || true
module load openmc/0.15.3-foss-2023a

source "$PWR_OPENMC_VENV/bin/activate"

export OPENMC_CROSS_SECTIONS

export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"

unset \
  OMP_PROC_BIND \
  OMP_PLACES \
  GOMP_CPU_AFFINITY


echo "============================================================"
echo "CYCLE5 q5 PROSPECTIVE360 — FRESH WARMSTART SHARD"
echo "============================================================"
echo "shard                 = $SHARD"
echo "output                = $SHARD_OUT"
echo "firewall              = $CYCLE5_Q5_PROSPECTIVE_FIREWALL"
echo "expected D4 excluded  = $CYCLE5_Q5_PROSPECTIVE_EXPECTED_D4"
echo "base seed             = $CYCLE5_Q5_PROSPECTIVE_WARMSTART_BASE_SEED"
echo "============================================================"


srun -n 1 -c 48 --cpu-bind=none \
python "$PWR_PROJECT/code/annealing/sample_loading_patterns.py" \
  --shard-id "$SHARD" \
  --output-dir "$SHARD_OUT" \
  --exclude-pattern-file "$PWR_PROJECT/run/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360/protocol/cycle5_q5_prospective360_generator_exact_unique_exclusion_v1.json" \
  --expected-excluded 32333 \
  --samples-per-group 10 \
  --workers 48 \
  --max-batches-per-group 5000 \
  --base-seed "$CYCLE5_Q5_PROSPECTIVE_WARMSTART_BASE_SEED" \
  --resume
