#!/bin/bash
#SBATCH --job-name=final_tail10k
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=run/logs/final_tail10k_%j.out
#SBATCH --error=run/logs/final_tail10k_%j.err
set -euo pipefail
PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
INPUT="$PWR_RUN/labeled/round_4_labeled_17500.json"
cd "$PWR_PROJECT"
module purge || true
module load python/3.11.5-gcccore-13.2.0
source "$PWR_ML_VENV/bin/activate"
export PYTHONPATH="$PWR_SCRIPTS:${PYTHONPATH:-}"
python "$PWR_SCRIPTS/data/select_optimizer_tail.py" \
  --labeled-17500 "$INPUT" \
  --diversity-model "$PWR_DIVERSITY_MODEL" \
  --output "$PWR_FINAL_10000" \
  --target 10000 \
  --diversity-fraction-of-nonfeasible-fill 0.20
python "$PWR_SCRIPTS/data/check_evaluation_data.py" --dataset "$PWR_FINAL_10000" --expected 10000
