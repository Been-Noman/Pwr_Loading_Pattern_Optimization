#!/bin/bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^[1-4]$ ]]; then
  echo "Usage: $0 <wave 1|2|3|4>" >&2
  exit 2
fi
WAVE="$1"
PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
cd "$PWR_PROJECT"
[[ -f "$PWR_PROJECT/ml_preflight_report.json" ]] || { echo "Missing ml_preflight_report.json; run and pass hpc/jobs/active_learning/01_ml_preflight.sbatch first." >&2; exit 4; }
grep -q '"status": "PASS"' "$PWR_PROJECT/ml_preflight_report.json" || { echo "ML preflight did not report PASS." >&2; exit 4; }

case "$WAVE" in
  1) PREV_COUNT=7500;  NEW_COUNT=10000 ;;
  2) PREV_COUNT=10000; NEW_COUNT=12500 ;;
  3) PREV_COUNT=12500; NEW_COUNT=15000 ;;
  4) PREV_COUNT=15000; NEW_COUNT=17500 ;;
esac
PREV_ROUND=$((WAVE-1))
LABELED="$PWR_RUN/labeled/round_${PREV_ROUND}_labeled_${PREV_COUNT}.json"
OUTPUT_LABELED="$PWR_RUN/labeled/round_${WAVE}_labeled_${NEW_COUNT}.json"
ROUND_DIR="$PWR_RUN/cnn/round_${PREV_ROUND}_${PREV_COUNT}"
SCORE_DIR="$ROUND_DIR/scores_for_wave_${WAVE}"

[[ -f "$LABELED" ]] || { echo "Missing labelled dataset: $LABELED" >&2; exit 3; }
[[ -f "$PWR_CANDIDATE_BANK" ]] || { echo "Missing candidate bank: $PWR_CANDIDATE_BANK" >&2; exit 3; }

PREP=$(sbatch --parsable \
  --export=ALL,DATASET_PATH="$LABELED",ROUND_DIR="$ROUND_DIR",ROUND_NAME="active_round_${PREV_ROUND}_${PREV_COUNT}" \
  "$PWR_PROJECT/hpc/jobs/active_learning/05_prepare_cnn_round.sbatch")
TRAIN=$(sbatch --parsable --dependency=afterok:$PREP \
  --export=ALL,ROUND_DIR="$ROUND_DIR" \
  "$PWR_PROJECT/hpc/jobs/active_learning/06_train_cnn_5seeds.sbatch")
SCORE=$(sbatch --parsable --dependency=afterok:$TRAIN \
  --export=ALL,ROUND_DIR="$ROUND_DIR",SCORE_DIR="$SCORE_DIR" \
  "$PWR_PROJECT/hpc/jobs/active_learning/07_score_candidate_bank.sbatch")
SELECT=$(sbatch --parsable --dependency=afterok:$SCORE \
  --export=ALL,WAVE="$WAVE",LABELED_PATH="$LABELED",SCORE_DIR="$SCORE_DIR" \
  "$PWR_PROJECT/hpc/jobs/active_learning/08_select_wave_2500.sbatch")
OPENMC=$(sbatch --parsable --dependency=afterok:$SELECT \
  --export=ALL,WAVE="$WAVE" \
  "$PWR_PROJECT/hpc/jobs/active_learning/09_openmc_wave_2500.sbatch")
MERGE=$(sbatch --parsable --dependency=afterok:$OPENMC \
  --export=ALL,WAVE="$WAVE",PREVIOUS_LABELED="$LABELED",OUTPUT_LABELED="$OUTPUT_LABELED" \
  "$PWR_PROJECT/hpc/jobs/active_learning/10_merge_openmc_wave.sbatch")

cat <<EOF
Submitted active-learning Wave $WAVE:
  CNN dataset:       $LABELED ($PREV_COUNT rows)
  prepare CNN:       $PREP
  train 5 seeds:     $TRAIN
  score 50k bank:    $SCORE
  select 2,500:      $SELECT
  OpenMC 100 nodes:  $OPENMC
  validate + merge:  $MERGE
  next dataset:      $OUTPUT_LABELED ($NEW_COUNT rows)

Do not submit the next wave until $MERGE succeeds and its wave_validation_summary.json is inspected.
EOF
