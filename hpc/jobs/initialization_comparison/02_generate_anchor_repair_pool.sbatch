#!/bin/bash
#SBATCH --job-name=anchor_repair
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=run/logs/formal_anchor_repair_%j.out
#SBATCH --error=run/logs/formal_anchor_repair_%j.err

set -euo pipefail

PROJECT="${PWR_PROJECT:-$SCRATCH/pwr_active_learning}"
source "$PROJECT/config.sh"

OUT="$PWR_RUN/formal_random_sa_sqa/manifests/anchor_repair_pool"
mkdir -p "$OUT"

cd "$PWR_PROJECT"

module purge || true
module load openmc/0.15.3-foss-2023a
source "$PWR_OPENMC_VENV/bin/activate"

export OPENMC_CROSS_SECTIONS
export PYTHONPATH="$PWR_PIPELINE_DIR:$PWR_SCRIPTS:${PYTHONPATH:-}"

unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY

srun -n 1 -c 48 --cpu-bind=none \
python "$PWR_SCRIPTS/annealing/sample_loading_patterns.py" \
  --shard-id 1000 \
  --output-dir "$OUT" \
  --exclude-pattern-file \
    "$PWR_RUN/formal_random_sa_sqa/manifests/repair_exact_exclusion.json" \
  --expected-excluded 21996 \
  --samples-per-group 30 \
  --workers 48 \
  --max-batches-per-group 5000 \
  --base-seed 20260816 \
  --resume
