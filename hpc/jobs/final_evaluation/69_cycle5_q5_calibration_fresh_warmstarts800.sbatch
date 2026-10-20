#!/bin/bash
#SBATCH --job-name=c5q5warm800
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --array=0-799%100
#SBATCH --output=run/logs/c5q5warm800_%A_%a.out
#SBATCH --error=run/logs/c5q5warm800_%A_%a.err

set -euo pipefail

PROJECT="$PWR_PROJECT"

source "$PROJECT/config.sh"

SHARD="$SLURM_ARRAY_TASK_ID"

OUT="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_calibration/warmstarts/parallel100/raw/shard_${SHARD}"

FIREWALL="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_calibration/protocol/cycle5_q5_known_openmc_d4_exclusion_31386_v1.json"

DESIGN="$PWR_RUN/optimizer_tail_v2/cycle5_balanced_multiarm/q5_calibration/protocol/cycle5_q5_warmstart_generation_design_v1.json"

GENERATOR="$PWR_PROJECT/code/annealing/sample_loading_patterns.py"

[[ "$(sha256sum "$FIREWALL" | awk '{print $1}')" == "ab483ce07e05b22a657f2f0b6e5caac03f782d24f3928e6da9e6dadd538de792" ]]
[[ "$(sha256sum "$DESIGN" | awk '{print $1}')" == "b7c4c2bb384cbf317ac594b3330bc737e7e60681226748ab983adc88a83b2120" ]]
[[ "$(sha256sum "$GENERATOR" | awk '{print $1}')" == "14440a3e72594ffd3befacb5067fa84013949bafc2d580a3f857328c02bf9d21" ]]

mkdir -p "$OUT" "$PWR_RUN/logs"

cd "$PWR_PROJECT"

module purge || true
module load openmc/0.15.3-foss-2023a

source "$PWR_OPENMC_VENV/bin/activate"

export OPENMC_CROSS_SECTIONS
export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"

unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY

echo "============================================================"
echo "CYCLE5 q5 CALIBRATION FRESH WARMSTART800"
echo "SHARD=$SHARD"
echo "REPLICATE=$((SHARD / 10))"
echo "LOCAL_SHARD=$((SHARD % 10))"
echo "KNOWN_OPENMC_D4=31386"
echo "SAMPLES_PER_GROUP=10"
echo "WORKERS=48"
echo "BASE_SEED=2026082811"
echo "OPENMC=NO"
echo "Q5_FIT=NO"
echo "============================================================"

srun     -n 1     -c 48     --cpu-bind=none     python "$GENERATOR"     --shard-id "$SHARD"     --output-dir "$OUT"     --exclude-pattern-file "$FIREWALL"     --expected-excluded 31386     --samples-per-group 10     --workers 48     --max-batches-per-group 5000     --base-seed 2026082811     --resume

echo "CYCLE5_Q5_WARMSTART_SHARD_${SHARD}_DONE"
