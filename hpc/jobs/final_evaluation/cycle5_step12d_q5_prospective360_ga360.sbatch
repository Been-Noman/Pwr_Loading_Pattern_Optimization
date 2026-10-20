#!/bin/bash -l
#SBATCH --job-name=q5prosga360
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --array=0-359%120
#SBATCH --output=run/logs/q5prosga360_%A_%a.out
#SBATCH --error=run/logs/q5prosga360_%A_%a.err

set -euo pipefail

cd $PWR_PROJECT
source config.sh

module purge || true
module load python/3.11.5-gcccore-13.2.0

source "$PWR_ML_VENV/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTEROP_THREADS=4
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

export PYTHONPATH="$PWR_PROJECT/code/optimization:$PWR_SCRIPTS:${PYTHONPATH:-}"

python code/optimization/run_paired_search_task.py \
  --manifest run/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360/protocol/cycle5_q5_prospective360_execution_manifest_360_v1.json \
  --expected-manifest-sha 76f8e5c0598bd8a9312a2c49c41a3177c9d01b9518fe3bb8bfc95ad0b8baaee9 \
  --task-id "${SLURM_ARRAY_TASK_ID}"
