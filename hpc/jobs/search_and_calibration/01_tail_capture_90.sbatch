#!/bin/bash
#SBATCH --job-name=v2_tail
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --array=0-89%90
#SBATCH --output=run/logs/v2_tail_%A_%a.out
#SBATCH --error=run/logs/v2_tail_%A_%a.err

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

TASK="$SLURM_ARRAY_TASK_ID"
REPLICATE=$(( TASK / 3 ))
ARM=$(( TASK % 3 ))

case "$ARM" in
    0)
        STRATEGY="random"
        OUTPUT_DIR="$PWR_RUN/optimizer_tail_v2/candidate_bank/raw/random"
        ;;
    1)
        STRATEGY="sa"
        OUTPUT_DIR="$PWR_RUN/optimizer_tail_v2/candidate_bank/raw/sa"
        ANCHOR_JSON="$PWR_RUN/formal_random_sa_sqa/manifests/anchor_banks/sa/anchors_$(printf '%02d' "$REPLICATE").json"
        ;;
    2)
        STRATEGY="sqa"
        OUTPUT_DIR="$PWR_RUN/optimizer_tail_v2/candidate_bank/raw/sqa"
        ANCHOR_JSON="$PWR_RUN/formal_random_sa_sqa/manifests/anchor_banks/sqa/anchors_$(printf '%02d' "$REPLICATE").json"
        ;;
esac

mkdir -p "$OUTPUT_DIR"

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
    python "$PWR_SCRIPTS/optimization/collect_optimization_tail.py" \
    "${ARGS[@]}"
