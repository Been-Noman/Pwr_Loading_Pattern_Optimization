#!/bin/bash
#SBATCH --job-name=sa-sqa-pattern2500
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=external_openmc_%x_%j.out
#SBATCH --error=external_openmc_%x_%j.err
set -euo pipefail

SCRATCH_ROOT="${SCRATCH:-$HOME}"
BASE="$SCRATCH_ROOT/openmc2500"
SRC="${PWR_PROJECT:?Set PWR_PROJECT to the repository root}"
VENV="$SCRATCH_ROOT/venvs/openmc-sa-sqa"
XS="$SCRATCH_ROOT/openmc_data/endfb-vii.1-hdf5/cross_sections.xml"
RUN="$BASE/outputs/production_sa_sqa_2500"
mkdir -p "$BASE/outputs" "$RUN" "$RUN/chunks" "$RUN/metrics" "$RUN/tmp"

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
trap 'rm -rf "$JOBLIB_TEMP_FOLDER"' EXIT

srun -n 1 -c 48 --cpu-bind=none python "$SRC/code/annealing/qubo_and_reactor_model.py" solver \
  --output-dir "$RUN" \
  --pattern-file "$RUN/patterns_sa_sqa_2500.json" \
  --samples-per-group 625 \
  --sa-workers 48 \
  --sqa-workers 48 \
  --max-batches-per-group 25000 \
  --base-seed 20260718 \
  --resume

python - <<'PY_CHECK'
import json, os
from collections import Counter
from pathlib import Path
root = Path(os.environ.get('SCRATCH', str(Path.home())))
p = root/'openmc2500/outputs/production_sa_sqa_2500/patterns_sa_sqa_2500.json'
rows = json.loads(p.read_text())
assert len(rows) == 2500, len(rows)
assert [int(r['id']) for r in rows] == list(range(2500))
counts = Counter((r['solver'], bool(r['symmetry_enabled'])) for r in rows)
print('pattern bank counts:', counts)
assert set(counts.values()) == {625}, counts
PY_CHECK

echo "Production pattern bank completed at $(date --iso-8601=seconds)"
