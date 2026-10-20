#!/bin/bash
#SBATCH --job-name=select2500
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=run/logs/select_wave_%j.out
#SBATCH --error=run/logs/select_wave_%j.err
set -euo pipefail
: "${WAVE:?export WAVE before sbatch}"
: "${LABELED_PATH:?export LABELED_PATH before sbatch}"
: "${SCORE_DIR:?export SCORE_DIR before sbatch}"
PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
WAVE_DIR="$PWR_RUN/waves/wave_${WAVE}"
mkdir -p "$WAVE_DIR"
cd "$PWR_PROJECT"
module purge || true
module load python/3.11.5-gcccore-13.2.0
source "$PWR_ML_VENV/bin/activate"
export PYTHONPATH="$PWR_SCRIPTS:${PYTHONPATH:-}"
python "$PWR_SCRIPTS/data/select_new_simulations.py" \
  --candidate-bank "$PWR_CANDIDATE_BANK" \
  --scores "$SCORE_DIR/candidate_ensemble_scores.csv" \
  --labeled-dataset "$LABELED_PATH" \
  --diversity-model "$PWR_DIVERSITY_MODEL" \
  --history "$PWR_SELECTION_HISTORY" \
  --wave "$WAVE" \
  --output "$WAVE_DIR/selected_2500.json"
