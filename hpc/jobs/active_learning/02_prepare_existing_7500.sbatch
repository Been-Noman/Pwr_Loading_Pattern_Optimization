#!/bin/bash
#SBATCH --job-name=prep_old7500
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=run/logs/prep_old7500_%j.out
#SBATCH --error=run/logs/prep_old7500_%j.err
set -euo pipefail
PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
cd "$PWR_PROJECT"
module purge || true
module load python/3.11.5-gcccore-13.2.0
source "$PWR_ML_VENV/bin/activate"
export PYTHONPATH="$PWR_SCRIPTS:${PYTHONPATH:-}"
python "$PWR_SCRIPTS/data/prepare_initial_dataset.py" \
  --input "$PWR_EXISTING_2500" \
  --input "$PWR_EXISTING_5000" \
  --output "$PWR_EXISTING_7500" \
  --patterns-output "$PWR_EXCLUSION_7500" \
  --expected 7500
