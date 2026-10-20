#!/usr/bin/env bash
# Copy to config.sh (gitignored), set your site's paths, then source it.
# PWR_PROJECT should point to the directory containing this file.
: "${SCRATCH:?Set SCRATCH to a writable scratch directory}"
export PWR_PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
export PWR_RUN="$PWR_PROJECT/run"
export PWR_SCRIPTS="$PWR_PROJECT/code"
export PWR_SLURM="$PWR_PROJECT/hpc/jobs"

# Environments are intentionally distinct: TensorFlow/CNN vs OpenMC/SA/SQA.
export PWR_ML_VENV="${PWR_ML_VENV:-$SCRATCH/venvs/pwr-cnn-active}"
export PWR_OPENMC_VENV="${PWR_OPENMC_VENV:-$SCRATCH/venvs/openmc-sa-sqa}"

# Original SA/SQA and OpenMC source modules are grouped under code/.
export PWR_PIPELINE_DIR="${PWR_PIPELINE_DIR:-$PWR_PROJECT/code/annealing}"
export PWR_PIPELINE_MODULE="$PWR_PIPELINE_DIR/qubo_and_reactor_model.py"
export PWR_OPENMC_RUNNER="$PWR_PROJECT/code/physics/simulate_patterns.py"
export OPENMC_CROSS_SECTIONS="${OPENMC_CROSS_SECTIONS:-/path/to/openmc/cross_sections.xml}"

# Two PRE-EXISTING constrained data sets; bring them separately.
export PWR_EXISTING_2500="${PWR_EXISTING_2500:-/path/to/pwr_sa_sqa_openmc_2500_merged.json}"
export PWR_EXISTING_5000="${PWR_EXISTING_5000:-/path/to/pwr_sa_sqa_openmc_new5000_merged.json}"

export PWR_EXISTING_7500="$PWR_RUN/labeled/round_0_labeled_7500.json"
export PWR_EXCLUSION_7500="$PWR_RUN/exclusions/existing_7500_patterns.json"
export PWR_SHARD_ROOT="$PWR_RUN/candidate_generation/shards"
export PWR_CANDIDATE_BANK="$PWR_RUN/candidate_bank/candidate_bank_50000.json"
export PWR_DIVERSITY_MODEL="$PWR_RUN/candidate_bank/diversity_model.joblib"
export PWR_SELECTION_HISTORY="$PWR_RUN/selection_history.json"
export PWR_FINAL_10000="$PWR_RUN/final/final_tail_10000.json"

mkdir -p "$PWR_RUN" "$PWR_RUN/labeled" "$PWR_RUN/exclusions" \
    "$PWR_RUN/candidate_generation" "$PWR_RUN/candidate_bank" \
    "$PWR_RUN/cnn" "$PWR_RUN/waves" "$PWR_RUN/final" "$PWR_RUN/logs"

# Import shared geometry, CNN, annealing and local-search modules from their
# scientifically named source directories. Preserve existing site PYTHONPATH.
export PYTHONPATH="$PWR_SCRIPTS/core:$PWR_SCRIPTS/annealing:$PWR_SCRIPTS/physics:$PWR_SCRIPTS/data:$PWR_SCRIPTS/surrogate:$PWR_SCRIPTS/uncertainty:$PWR_SCRIPTS/optimization:$PWR_SCRIPTS/evaluation:${PYTHONPATH:-}"
