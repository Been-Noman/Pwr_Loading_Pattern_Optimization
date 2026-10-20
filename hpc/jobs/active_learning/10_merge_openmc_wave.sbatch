#!/bin/bash
#SBATCH --job-name=merge_wave
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=run/logs/merge_wave_%j.out
#SBATCH --error=run/logs/merge_wave_%j.err
set -euo pipefail
: "${WAVE:?export WAVE before sbatch}"
: "${PREVIOUS_LABELED:?export PREVIOUS_LABELED before sbatch}"
: "${OUTPUT_LABELED:?export OUTPUT_LABELED before sbatch}"
PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
RUN="$PWR_RUN/waves/wave_${WAVE}"
cd "$PWR_PROJECT"
module purge || true
module load python/3.11.5-gcccore-13.2.0
source "$PWR_ML_VENV/bin/activate"
export PYTHONPATH="$PWR_SCRIPTS:${PYTHONPATH:-}"
python "$PWR_SCRIPTS/physics/merge_simulation_results.py" \
  --wave-dir "$RUN" \
  --selected-patterns "$RUN/selected_2500.json" \
  --previous-labeled "$PREVIOUS_LABELED" \
  --output-labeled "$OUTPUT_LABELED" \
  --expected-wave 2500 --chunk-size 25 --wave "$WAVE"
