#!/bin/bash
set -euo pipefail
SCRATCH_ROOT="${SCRATCH:-$HOME}"
BASE="$SCRATCH_ROOT/openmc2500"
SRC="${PWR_PROJECT:?Set PWR_PROJECT to the repository root}"
VENV="$SCRATCH_ROOT/venvs/openmc-sa-sqa"
XS="$SCRATCH_ROOT/openmc_data/endfb-vii.1-hdf5/cross_sections.xml"

module purge
module load openmc/0.15.3-foss-2023a
source "$VENV/bin/activate"
export OPENMC_CROSS_SECTIONS="$XS"

echo "host=$(hostname)"
echo "python=$(command -v python)"
echo "openmc=$(command -v openmc)"
echo "cross_sections=$OPENMC_CROSS_SECTIONS"
echo "scratch=$SCRATCH_ROOT"

test -x "$VENV/bin/python"
test -r "$XS"
command -v openmc >/dev/null
command -v taskset >/dev/null

python - <<'PY_CHECK'
import os, sys
from pathlib import Path
for name in ['openmc','numpy','h5py','tqdm','joblib','dimod','openjij']:
    mod = __import__(name)
    print(f'{name}: version={getattr(mod, "__version__", "unknown")} path={getattr(mod, "__file__", "unknown")}')
print('python:', sys.executable)
print('cross sections readable:', Path(os.environ['OPENMC_CROSS_SECTIONS']).is_file())
PY_CHECK

python -m py_compile \
  "$SRC/code/annealing/qubo_and_reactor_model.py" \
  "$SRC/code/physics/simulate_patterns.py" \
  "$SRC/code/physics/validate_reference_batch.py" \
  "$SRC/code/physics/estimate_runtime.py" \
  "$SRC/code/physics/inspect_pilot_results.py"

echo "Environment and Python syntax checks passed."
