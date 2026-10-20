#!/bin/bash
#SBATCH --job-name=c5q5repair
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --array=300-339%40
#SBATCH --output=run/logs/c5q5repair_%A_%a.out
#SBATCH --error=run/logs/c5q5repair_%A_%a.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

SHARD="$SLURM_ARRAY_TASK_ID"

PROS="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_prospective360"

SHARD_OUT="$PROS/warmstarts/repair_pool/raw/shard_${SHARD}"

FIREWALL="$PROS/protocol/cycle5_q5_prospective360_generator_exact_unique_exclusion_v1.json"

DESIGN="$PROS/protocol/cycle5_q5_prospective360_repair_pool_design_v1.json"

EXPECTED_FIREWALL_SHA="092544e1d417eb588d92ee2dd51b10b82c2396861b327d89b3a2074063047f73"
EXPECTED_DESIGN_SHA="e29e90aa10c0f84bb6aac66baeb42c9559110b032bae8971639f16f167188627"

mkdir -p "$SHARD_OUT" "$PWR_RUN/logs"

ACTUAL_FIREWALL_SHA="$(sha256sum "$FIREWALL" | awk '{print $1}')"
ACTUAL_DESIGN_SHA="$(sha256sum "$DESIGN" | awk '{print $1}')"

if [[ "$ACTUAL_FIREWALL_SHA" != "$EXPECTED_FIREWALL_SHA" ]]; then
    echo "ERROR: prospective generator firewall changed" >&2
    exit 2
fi

if [[ "$ACTUAL_DESIGN_SHA" != "$EXPECTED_DESIGN_SHA" ]]; then
    echo "ERROR: frozen repair-pool design changed" >&2
    exit 3
fi

cd "$PWR_PROJECT"

module purge || true
module load openmc/0.15.3-foss-2023a
source "$PWR_OPENMC_VENV/bin/activate"

export OPENMC_CROSS_SECTIONS
export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"

unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY

echo "============================================================"
echo "CYCLE5 q5 PROSPECTIVE360 REPAIR40"
echo "SHARD=$SHARD"
echo "NODE=$HOSTNAME"
echo "FIREWALL_SHA=$ACTUAL_FIREWALL_SHA"
echo "DESIGN_SHA=$ACTUAL_DESIGN_SHA"
echo "EXPECTED_EXCLUDED=32333"
echo "SAMPLES_PER_GROUP=10"
echo "WORKERS=48"
echo "MAX_BATCHES_PER_GROUP=5000"
echo "BASE_SEED=2026082911"
echo "============================================================"

srun -n 1 -c 48 --cpu-bind=none \
python "$PWR_PROJECT/code/annealing/sample_loading_patterns.py" \
  --shard-id "$SHARD" \
  --output-dir "$SHARD_OUT" \
  --exclude-pattern-file "$FIREWALL" \
  --expected-excluded 32333 \
  --samples-per-group 10 \
  --workers 48 \
  --max-batches-per-group 5000 \
  --base-seed 2026082911 \
  --resume

echo "CYCLE5_Q5_REPAIR_SHARD_${SHARD}_DONE"
