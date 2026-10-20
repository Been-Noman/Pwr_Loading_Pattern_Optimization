#!/bin/bash
set -euo pipefail
PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"
cd "$PWR_PROJECT"
[[ -f "$PWR_PROJECT/ml_preflight_report.json" ]] || { echo "Missing ml_preflight_report.json; run and pass hpc/jobs/active_learning/01_ml_preflight.sbatch first." >&2; exit 4; }
grep -q '"status": "PASS"' "$PWR_PROJECT/ml_preflight_report.json" || { echo "ML preflight did not report PASS." >&2; exit 4; }

PREP=$(sbatch --parsable "$PWR_PROJECT/hpc/jobs/active_learning/02_prepare_existing_7500.sbatch")
GEN=$(sbatch --parsable --dependency=afterok:$PREP "$PWR_PROJECT/hpc/jobs/active_learning/03_generate_80k_shards.sbatch")
BANK=$(sbatch --parsable --dependency=afterok:$GEN "$PWR_PROJECT/hpc/jobs/active_learning/04_build_candidate_bank_50k.sbatch")
cat <<EOF
Submitted candidate-generation chain:
  prepare existing 7500: $PREP
  generate ~80k raw:    $GEN
  build 50k bank:       $BANK
Final job to watch: $BANK
EOF
