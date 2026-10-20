#!/bin/bash
#SBATCH --job-name=c5resid5
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --output=run/logs/c5resid5_%j.out
#SBATCH --error=run/logs/c5resid5_%j.err

set -euo pipefail

PROJECT="$PWR_PROJECT"

source "$PROJECT/config.sh"

cd "$PROJECT"

OOF="$PWR_RUN/optimizer_tail_v2/crossfit_round5/oof/crossfit_round5_oof_predictions.npz"

AUDIT="$PWR_RUN/optimizer_tail_v2/crossfit_round5/oof/crossfit_round5_oof_audit.json"

OUT="$PWR_RUN/optimizer_tail_v2/models/learned_residual_safety_round5"

[[ "$(sha256sum "$OOF" | awk '{print $1}')" == "c963028d84d0b5cb5fe710c27b2ed07ab181df749fe4bac9cc85212b5e13a72f" ]]

[[ "$(sha256sum "$AUDIT" | awk '{print $1}')" == "4f2881be9435881568ea39e2b2c1544da73430c8daa427e63d702bd8976a4f48" ]]

[[ "$(sha256sum "$PWR_PROJECT/code/uncertainty/train_final_error_scales.py" | awk '{print $1}')" == "58a74918a6b5bd2fdb1c798edfdc0c7076378385aaba15a6e8cb9f123c9f8aa5" ]]

module purge || true
module load python/3.11.5-gcccore-13.2.0

source "$PWR_ML_VENV/bin/activate"

export PYTHONPATH="$PWR_SCRIPTS:${PYTHONPATH:-}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "============================================================"
echo "CYCLE5 RESIDUAL5"
echo "OOF_ROWS=10250"
echo "FEATURES=202"
echo "SECOND_LEVEL_OOF_FOLDS=5"
echo "TRAIN_PER_FOLD=8200"
echo "HOLDOUT_PER_FOLD=2050"
echo "DIAGNOSTIC_TREES=750"
echo "PRODUCTION_TREES=1000"
echo "MIN_SAMPLES_LEAF=5"
echo "MAX_FEATURES=0.7"
echo "LOCKED600_USED=NO"
echo "Q5_CALIBRATED=NO"
echo "============================================================"

srun     -n 1     -c "$SLURM_CPUS_PER_TASK"     --cpu-bind=none     python "$PWR_PROJECT/code/uncertainty/train_final_error_scales.py"     --oof "$OOF"     --oof-audit "$AUDIT"     --output-dir "$OUT"

echo "CYCLE5_RESIDUAL5_TRAINING_DONE"
