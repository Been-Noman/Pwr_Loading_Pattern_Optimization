#!/bin/bash
#SBATCH --job-name=v2rga_ss
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=run/logs/v2rga_ss_%A_%a.out
#SBATCH --error=run/logs/v2rga_ss_%A_%a.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

TASK="$SLURM_ARRAY_TASK_ID"

case "$TASK" in
    0)
        STRATEGY="sa"
        ANCHOR_JSON="$PWR_RUN/formal_random_sa_sqa/manifests/anchor_banks/sa/anchors_00.json"
        ;;
    1)
        STRATEGY="sqa"
        ANCHOR_JSON="$PWR_RUN/formal_random_sa_sqa/manifests/anchor_banks/sqa/anchors_00.json"
        ;;
esac

OUT="$PWR_RUN/optimizer_tail_v2/robust_ga_smoke/$STRATEGY"

mkdir -p "$OUT" "$PWR_RUN/logs"

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

echo "TASK=$TASK STRATEGY=$STRATEGY ANCHOR=$ANCHOR_JSON"

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=none \
  python "$PWR_PROJECT/code/optimization/penalty_based_search.py" \
  --round-dir "$PWR_RUN/optimizer_tail_v2/models/round2_cnn" \
  --residual-model-dir "$PWR_RUN/optimizer_tail_v2/models/learned_residual_safety" \
  --output-dir "$OUT" \
  --strategy "$STRATEGY" \
  --anchor-json "$ANCHOR_JSON" \
  --labelled-json "$PWR_RUN/optimizer_tail_v2/manifests/known_openmc_all_d4_exclusion.json" \
  --seed-index 0 \
  --threads "$SLURM_CPUS_PER_TASK"
