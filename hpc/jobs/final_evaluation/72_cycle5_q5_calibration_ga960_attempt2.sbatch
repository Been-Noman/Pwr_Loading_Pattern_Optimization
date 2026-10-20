#!/bin/bash -l
#SBATCH --job-name=c5q5ga960a2
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --array=0-959%240
#SBATCH --output=run/logs/c5q5ga960a2_%A_%a.out
#SBATCH --error=run/logs/c5q5ga960a2_%A_%a.err

set -euo pipefail

cd "$PWR_PROJECT"

source config.sh

module purge || true
module load python/3.11.5-gcccore-13.2.0

source "$PWR_ML_VENV/bin/activate"

export PYTHONPATH="$PWR_SCRIPTS:${PYTHONPATH:-}"

MANIFEST="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_calibration/protocol/cycle5_q5_calibration_execution_manifest_960_attempt2_v1.json"

DISPATCHER="$PWR_PROJECT/code/optimization/run_paired_search_task.py"

ADAPTER="$PWR_PROJECT/code/optimization/run_balanced_calibration_search.py"

[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "797cff006d1f7b9686ce861d9271ca7ec7749d8dd385ebef10a1e27479f44bef" ]]
[[ "$(sha256sum "$DISPATCHER" | awk '{print $1}')" == "0dbac90d9827a757bcbc5b15982c46ecf46ad0152f4794a634da12be6181a51c" ]]
[[ "$(sha256sum "$ADAPTER" | awk '{print $1}')" == "74bba0f2fc2f61e1733f437d0c6c50bb7df3684f4ab9787d2020b6a44fe6dd82" ]]

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python "$DISPATCHER"     --manifest "$MANIFEST"     --expected-manifest-sha "797cff006d1f7b9686ce861d9271ca7ec7749d8dd385ebef10a1e27479f44bef"     --task-id "$SLURM_ARRAY_TASK_ID"
