#!/bin/bash
#SBATCH --job-name=fair4x120
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --array=0-119
#SBATCH --output=run/logs/fair4x120_%A_%a.out
#SBATCH --error=run/logs/fair4x120_%A_%a.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

TASK="$SLURM_ARRAY_TASK_ID"

# ============================================================
# 120 independent runs:
#
# 4 solvers × 10 replicates × 3 initialization strategies
#
# Solver block:
#   0-29   V2-300
#   30-59  V3-300
#   60-89  V2M-300
#   90-119 V3M-300
#
# Within each 30-run block:
#   replicate = 0..9
#   arm 0 = Random
#   arm 1 = SA
#   arm 2 = SQA
# ============================================================

SOLVER_INDEX=$(( TASK / 30 ))
INNER_TASK=$(( TASK % 30 ))

REPLICATE=$(( INNER_TASK / 3 ))
ARM=$(( INNER_TASK % 3 ))

case "$SOLVER_INDEX" in
    0)
        SOLVER="v2_300"
        RUNNER="$PWR_PROJECT/code/optimization/penalty_based_search_300.py"
        ;;
    1)
        SOLVER="v3_300"
        RUNNER="$PWR_PROJECT/code/optimization/feasibility_first_search.py"
        ;;
    2)
        SOLVER="v2m_300"
        RUNNER="$PWR_PROJECT/code/optimization/penalty_based_memetic_search.py"
        ;;
    3)
        SOLVER="v3m_300"
        RUNNER="$PWR_PROJECT/code/optimization/feasibility_first_memetic_search.py"
        ;;
    *)
        echo "Invalid SOLVER_INDEX=$SOLVER_INDEX" >&2
        exit 2
        ;;
esac

case "$ARM" in
    0)
        STRATEGY="random"
        ;;
    1)
        STRATEGY="sa"
        ANCHOR_JSON="$PWR_RUN/formal_random_sa_sqa/manifests/anchor_banks/sa/anchors_$(printf '%02d' "$REPLICATE").json"
        ;;
    2)
        STRATEGY="sqa"
        ANCHOR_JSON="$PWR_RUN/formal_random_sa_sqa/manifests/anchor_banks/sqa/anchors_$(printf '%02d' "$REPLICATE").json"
        ;;
    *)
        echo "Invalid ARM=$ARM" >&2
        exit 2
        ;;
esac

OUT="$PWR_RUN/optimizer_tail_v2/fair_comparison120/$SOLVER/$STRATEGY"

mkdir -p "$OUT" "$PWR_RUN/logs"

cd "$PWR_PROJECT"

module purge || true
module load python/3.11.5-gcccore-13.2.0
source "$PWR_ML_VENV/bin/activate"

export PYTHONPATH="$PWR_SCRIPTS/optimization:$PWR_SCRIPTS:${PYTHONPATH:-}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTEROP_THREADS=4
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

ARGS=(
    --round-dir "$PWR_RUN/optimizer_tail_v2/models/round2_cnn"
    --residual-model-dir "$PWR_RUN/optimizer_tail_v2/models/learned_residual_safety"
    --output-dir "$OUT"
    --strategy "$STRATEGY"
    --labelled-json "$PWR_RUN/optimizer_tail_v2/manifests/known_openmc_all_d4_exclusion.json"
    --seed-index "$REPLICATE"
    --threads "$SLURM_CPUS_PER_TASK"
)

if [[ "$STRATEGY" != "random" ]]; then
    ARGS+=(--anchor-json "$ANCHOR_JSON")
fi

GA_SEED=$(( 2026 + 1009 * REPLICATE ))

echo "============================================================"
echo "FAIR 4-SOLVER BENCHMARK"
echo "ARRAY_TASK=$TASK"
echo "SOLVER_INDEX=$SOLVER_INDEX"
echo "SOLVER=$SOLVER"
echo "RUNNER=$RUNNER"
echo "INNER_TASK=$INNER_TASK"
echo "REPLICATE=$REPLICATE"
echo "GA_SEED=$GA_SEED"
echo "STRATEGY=$STRATEGY"
echo "NODE=$(hostname)"
echo "OUTPUT=$OUT"

if [[ "$STRATEGY" != "random" ]]; then
    echo "ANCHOR=$ANCHOR_JSON"
fi

echo "============================================================"

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=none \
    python "$RUNNER" \
    "${ARGS[@]}"
