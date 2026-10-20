#!/bin/bash
set -euo pipefail
PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
cd "$PWR_PROJECT"
[[ -f "$PWR_PROJECT/ml_preflight_report.json" ]] || { echo "Missing ml_preflight_report.json; run and pass hpc/jobs/active_learning/01_ml_preflight.sbatch first." >&2; exit 4; }
grep -q '"status": "PASS"' "$PWR_PROJECT/ml_preflight_report.json" || { echo "ML preflight did not report PASS." >&2; exit 4; }
INPUT="$PWR_RUN/labeled/round_4_labeled_17500.json"
[[ -f "$INPUT" ]] || { echo "Missing final cumulative dataset: $INPUT" >&2; exit 3; }

SELECT=$(sbatch --parsable "$PWR_PROJECT/hpc/jobs/active_learning/11_select_final_tail_10000.sbatch")
FINAL_ROUND="$PWR_RUN/cnn/final_tail_10000"
PREP=$(sbatch --parsable --dependency=afterok:$SELECT \
  --export=ALL,DATASET_PATH="$PWR_FINAL_10000",ROUND_DIR="$FINAL_ROUND",ROUND_NAME="final_tail_10000" \
  "$PWR_PROJECT/hpc/jobs/active_learning/05_prepare_cnn_round.sbatch")
TRAIN=$(sbatch --parsable --dependency=afterok:$PREP \
  --export=ALL,ROUND_DIR="$FINAL_ROUND" \
  "$PWR_PROJECT/hpc/jobs/active_learning/06_train_cnn_5seeds.sbatch")
cat <<EOF
Submitted final-dataset/model chain:
  select + audit final 10k: $SELECT
  final D4-safe split:       $PREP
  final 5-CNN training:      $TRAIN
Final models: $FINAL_ROUND/models/
EOF
