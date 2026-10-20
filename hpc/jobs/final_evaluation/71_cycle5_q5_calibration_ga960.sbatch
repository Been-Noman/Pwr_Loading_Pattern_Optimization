#!/bin/bash -l
#SBATCH --job-name=c5q5ga960
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --array=0-959%240
#SBATCH --output=run/logs/c5q5ga960_%A_%a.out
#SBATCH --error=run/logs/c5q5ga960_%A_%a.err

set -euo pipefail

cd "$PWR_PROJECT"

source config.sh

module purge || true
module load python/3.11.5-gcccore-13.2.0

source "$PWR_ML_VENV/bin/activate"

MANIFEST="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_calibration/protocol/cycle5_q5_calibration_execution_manifest_960_v1.json"

DISPATCHER="$PWR_PROJECT/code/optimization/run_paired_search_task.py"

ADAPTER="$PWR_PROJECT/code/optimization/run_frozen_search_adapter.py"

[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "30d2f370db4704b231a2c97303b27202131585855f8773cb88ee771425df8578" ]]

[[ "$(sha256sum "$DISPATCHER" | awk '{print $1}')" == "0dbac90d9827a757bcbc5b15982c46ecf46ad0152f4794a634da12be6181a51c" ]]

[[ "$(sha256sum "$ADAPTER" | awk '{print $1}')" == "322cf870aba87e77ec3e6fa5ff984b5106a0552a7008a10a095bd2b8590ccd03" ]]

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python "$DISPATCHER"     --manifest "$MANIFEST"     --expected-manifest-sha "30d2f370db4704b231a2c97303b27202131585855f8773cb88ee771425df8578"     --task-id "$SLURM_ARRAY_TASK_ID"
