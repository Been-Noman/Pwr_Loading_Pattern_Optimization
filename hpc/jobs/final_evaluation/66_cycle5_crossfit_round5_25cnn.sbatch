#!/bin/bash
#SBATCH --job-name=c5cf25
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --array=0-24%25
#SBATCH --output=run/logs/c5cf25_%A_%a.out
#SBATCH --error=run/logs/c5cf25_%A_%a.err

set -euo pipefail

PROJECT="$PWR_PROJECT"

source "$PROJECT/config.sh"

SEEDS=(
    42
    314
    2718
    31415
    161803
)

TASK="$SLURM_ARRAY_TASK_ID"

FOLD=$(( TASK / 5 ))
SEED_INDEX=$(( TASK % 5 ))

SEED="${SEEDS[$SEED_INDEX]}"

ROUND_DIR="$PWR_RUN/optimizer_tail_v2/crossfit_round5/folds/fold_${FOLD}"

FREEZE="$PWR_RUN/optimizer_tail_v2/crossfit_round5/crossfit_round5_dataset_freeze_manifest_v1.json"

cd "$PWR_PROJECT"

[[ "$(sha256sum "$FREEZE" | awk '{print $1}')" == "336e744bf288397d5298ec7408c512a4be8aac493e47d9f7144165635a0a7f44" ]]

[[ "$(sha256sum "$PWR_PROJECT/code/surrogate/train_final_out_of_fold_cnn.py" | awk '{print $1}')" == "2b07078106770cd46959ff73f2ebc146350cbe955efca25addf0649c8b2f70d9" ]]

module purge || true

module load python/3.11.5-gcccore-13.2.0

source "$PWR_ML_VENV/bin/activate"

export PYTHONPATH="$PWR_SCRIPTS:${PYTHONPATH:-}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTEROP_THREADS=4
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "============================================================"
echo "CYCLE5 STRICT OOF CNN"
echo "TASK=$TASK"
echo "FOLD=$FOLD"
echo "SEED=$SEED"
echo "ROUND_DIR=$ROUND_DIR"
echo "TRAIN_ROWS=14380"
echo "HOLDOUT=2050"
echo "SAMPLER=16 BROAD + 16 EQUAL-DOMAIN TAIL"
echo "============================================================"

srun     -n 1     -c "$SLURM_CPUS_PER_TASK"     --cpu-bind=none     python     "$PWR_PROJECT/code/surrogate/train_final_out_of_fold_cnn.py"     --round-dir "$ROUND_DIR"     --seed "$SEED"     --threads "$SLURM_CPUS_PER_TASK"

echo "CYCLE5_CROSSFIT_FOLD_${FOLD}_SEED_${SEED}_DONE"
