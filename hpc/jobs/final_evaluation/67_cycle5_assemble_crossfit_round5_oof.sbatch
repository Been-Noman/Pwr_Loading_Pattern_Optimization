#!/bin/bash
#SBATCH --job-name=c5oof10250
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=run/logs/c5oof10250_%j.out
#SBATCH --error=run/logs/c5oof10250_%j.err

set -euo pipefail

PROJECT="$PWR_PROJECT"

source "$PROJECT/config.sh"

cd "$PROJECT"

[[ "$(sha256sum "$PWR_PROJECT/code/uncertainty/assemble_final_crossvalidation.py" | awk '{print $1}')" == "02b3dc7fc27e81662c0f292d4b2465311086053205686e8c51d8d34b3dc372f0" ]]
[[ "$(sha256sum "$PWR_PROJECT/run/optimizer_tail_v2/crossfit_round5/crossfit_round5_dataset_freeze_manifest_v1.json" | awk '{print $1}')" == "336e744bf288397d5298ec7408c512a4be8aac493e47d9f7144165635a0a7f44" ]]
[[ "$(sha256sum "$PWR_PROJECT/run/optimizer_tail_v2/crossfit_round5/tail_crossfit_5fold_manifest.json" | awk '{print $1}')" == "b66b16aca6a540d6047f824c99720174efec4d6e8544c2c16129113f7ec24d79" ]]

module purge || true
module load python/3.11.5-gcccore-13.2.0

source "$PWR_ML_VENV/bin/activate"

export PYTHONPATH="$PWR_SCRIPTS:${PYTHONPATH:-}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTEROP_THREADS=4
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

srun     -n 1     -c "$SLURM_CPUS_PER_TASK"     --cpu-bind=none     python "$PWR_PROJECT/code/uncertainty/assemble_final_crossvalidation.py"     --fold-root         "$PWR_RUN/optimizer_tail_v2/crossfit_round5/folds"     --manifest         "$PWR_RUN/optimizer_tail_v2/crossfit_round5/tail_crossfit_5fold_manifest.json"     --output-dir         "$PWR_RUN/optimizer_tail_v2/crossfit_round5/oof"     --threads "$SLURM_CPUS_PER_TASK"
