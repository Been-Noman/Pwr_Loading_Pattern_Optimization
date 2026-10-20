#!/bin/bash
#SBATCH --job-name=formal_ga
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --array=0-89%30
#SBATCH --output=run/logs/formal_ga_%A_%a.out
#SBATCH --error=run/logs/formal_ga_%A_%a.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
cd "$PWR_PROJECT"

module purge || true
module load python/3.11.5-gcccore-13.2.0
source "$PWR_ML_VENV/bin/activate"

export PYTHONPATH="$PWR_SCRIPTS:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTEROP_THREADS=4
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# ------------------------------------------------------------
# Interleaved formal design:
#
# task 0 = replicate 00 Random
# task 1 = replicate 00 SA
# task 2 = replicate 00 SQA
# task 3 = replicate 01 Random
# task 4 = replicate 01 SA
# task 5 = replicate 01 SQA
# ...
#
# Thus every replicate uses the same GA seed-index across arms.
# ------------------------------------------------------------
TASK="$SLURM_ARRAY_TASK_ID"
REPLICATE=$(( TASK / 3 ))
ARM=$(( TASK % 3 ))

case "$ARM" in
    0)
        STRATEGY="random"
        OUTPUT_DIR="$PWR_RUN/formal_random_sa_sqa/random"
        ;;
    1)
        STRATEGY="sa"
        OUTPUT_DIR="$PWR_RUN/formal_random_sa_sqa/sa"
        ANCHOR_JSON="$PWR_RUN/formal_random_sa_sqa/manifests/anchor_banks/sa/anchors_$(printf '%02d' "$REPLICATE").json"
        ;;
    2)
        STRATEGY="sqa"
        OUTPUT_DIR="$PWR_RUN/formal_random_sa_sqa/sqa"
        ANCHOR_JSON="$PWR_RUN/formal_random_sa_sqa/manifests/anchor_banks/sqa/anchors_$(printf '%02d' "$REPLICATE").json"
        ;;
    *)
        echo "Invalid ARM=$ARM" >&2
        exit 1
        ;;
esac

mkdir -p "$OUTPUT_DIR"

echo "FORMAL_TASK=$TASK"
echo "REPLICATE=$REPLICATE"
echo "STRATEGY=$STRATEGY"
echo "OUTPUT_DIR=$OUTPUT_DIR"

ARGS=(
    --round-dir "$PWR_RUN/cnn/final_tail_10000"
    --audit-json "$PWR_RUN/cnn/final_tail_10000/final_evaluation/final_ensemble_audit.json"
    --output-dir "$OUTPUT_DIR"
    --strategy "$STRATEGY"
    --labelled-json "$PWR_RUN/final/final_tail_10000.json"
    --seed-index "$REPLICATE"
    --threads "$SLURM_CPUS_PER_TASK"
)

if [[ "$STRATEGY" != "random" ]]; then
    ARGS+=(--anchor-json "$ANCHOR_JSON")
fi

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=none \
    python "$PWR_SCRIPTS/optimization/run_paired_initialization_study.py" \
    "${ARGS[@]}"
