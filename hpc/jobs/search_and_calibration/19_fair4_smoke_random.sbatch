#!/bin/bash
#SBATCH --job-name=fair4smk
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=run/logs/fair4_smoke_%A_%a.out
#SBATCH --error=run/logs/fair4_smoke_%A_%a.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

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

case "$SLURM_ARRAY_TASK_ID" in
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
    echo "Invalid array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

OUT="$PWR_RUN/optimizer_tail_v2/fair_comparison_smoke/$SOLVER/random"
mkdir -p "$OUT" "$PWR_RUN/logs"

echo "FAIR COMPARISON SMOKE"
echo "SOLVER=$SOLVER"
echo "STRATEGY=random"
echo "SEED_INDEX=0"
echo "GA_SEED=2026"
echo "OUTPUT=$OUT"
echo "NODE=$(hostname)"

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=none \
  python "$RUNNER" \
  --round-dir "$PWR_RUN/optimizer_tail_v2/models/round2_cnn" \
  --residual-model-dir "$PWR_RUN/optimizer_tail_v2/models/learned_residual_safety" \
  --output-dir "$OUT" \
  --strategy random \
  --labelled-json "$PWR_RUN/optimizer_tail_v2/manifests/known_openmc_all_d4_exclusion.json" \
  --seed-index 0 \
  --threads "$SLURM_CPUS_PER_TASK"
