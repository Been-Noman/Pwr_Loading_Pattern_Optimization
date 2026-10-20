#!/bin/bash
#SBATCH --job-name=sa-sqa-dry-ares
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH --output=external_openmc_%x_%j.out
#SBATCH --error=external_openmc_%x_%j.err
set -euo pipefail

SCRATCH_ROOT="${SCRATCH:-$HOME}"
BASE="$SCRATCH_ROOT/openmc2500"
SRC="${PWR_PROJECT:?Set PWR_PROJECT to the repository root}"
VENV="$SCRATCH_ROOT/venvs/openmc-sa-sqa"
XS="$SCRATCH_ROOT/openmc_data/endfb-vii.1-hdf5/cross_sections.xml"
RUN="$BASE/outputs/dryrun_sa_sqa_ares"
mkdir -p "$BASE/outputs" "$RUN"

module purge
module load openmc/0.15.3-foss-2023a
source "$VENV/bin/activate"
unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY
export OPENMC_CROSS_SECTIONS="$XS"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$SRC"
export JOBLIB_TEMP_FOLDER="$RUN/joblib_${SLURM_JOB_ID}"
mkdir -p "$JOBLIB_TEMP_FOLDER"
TMPROOT="$RUN/tmp_openmc_${SLURM_JOB_ID}"

(
  while true; do
    echo "===== $(date --iso-8601=seconds) ====="
    free -h
    ps -u "$USER" -o pid,ppid,psr,pcpu,pmem,rss,etime,cmd --sort=-rss | head -25
    sleep 60
  done
) > "$RUN/resource_monitor_${SLURM_JOB_ID}.log" 2>&1 &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  rm -rf "$JOBLIB_TEMP_FOLDER" "$TMPROOT"
}
trap cleanup EXIT

echo "host=$(hostname) job=${SLURM_JOB_ID} cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
echo "python=$(command -v python) openmc=$(command -v openmc)"

srun -n 1 -c 48 --cpu-bind=none python "$SRC/code/annealing/qubo_and_reactor_model.py" solver \
  --output-dir "$RUN" \
  --pattern-file "$RUN/patterns_sa_sqa_dry40.json" \
  --samples-per-group 10 \
  --sa-workers 48 \
  --sqa-workers 48 \
  --max-batches-per-group 2000 \
  --base-seed 20260718 \
  --resume

srun -n 1 -c 48 --cpu-bind=none python "$SRC/code/physics/simulate_patterns.py" \
  --pattern-file "$RUN/patterns_sa_sqa_dry40.json" \
  --output-file "$RUN/openmc_dry40.jsonl" \
  --metrics-file "$RUN/openmc_dry40_metrics.json" \
  --start-index 0 \
  --n-samples 40 \
  --workers 5 \
  --threads-per-run 9 \
  --batches 50 \
  --inactive 10 \
  --particles 20000 \
  --tmp-root "$TMPROOT" \
  --group-waves

echo "Dry run completed at $(date --iso-8601=seconds)"
