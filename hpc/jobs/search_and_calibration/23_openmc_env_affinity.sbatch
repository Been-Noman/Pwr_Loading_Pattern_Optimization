#!/bin/bash
#SBATCH --job-name=omcaff48
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:05:00
#SBATCH --output=run/logs/omcaff48_%j.out
#SBATCH --error=run/logs/omcaff48_%j.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

module purge || true
module load openmc/0.15.3-foss-2023a
source "$PWR_OPENMC_VENV/bin/activate"

export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"
export OMP_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun -n 1 -c 48 --cpu-bind=none python - <<'PY'
import os

def show(label):
    a = sorted(os.sched_getaffinity(0))
    print(label, "count=", len(a), "affinity=", a, flush=True)

show("PYTHON_START")

import numpy
show("AFTER_NUMPY")

import openmc
show("AFTER_OPENMC")

import qubo_and_reactor_model
show("AFTER_PIPELINE_IMPORT")
PY
