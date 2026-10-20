#!/bin/bash
set -euo pipefail

# Run this INSIDE an Ares compute-node allocation, not on login01/login02.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: run this inside a compute-node allocation (srun --pty ...)." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${PWR_ML_VENV:-$SCRATCH/venvs/pwr-cnn-active}"

module purge || true
module load python/3.11.5-gcccore-13.2.0
python3 --version

mkdir -p "$(dirname "$VENV")"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python3 -m pip --require-virtualenv --no-cache-dir install --upgrade pip setuptools wheel
python3 -m pip --require-virtualenv --no-cache-dir install -r "$SCRIPT_DIR/requirements-cnn.txt"
python3 -m pip freeze > "$SCRIPT_DIR/requirements-cnn.lock.txt"

echo "Created/updated ML environment: $VENV"
echo "PyTorch is intentionally NOT installed: the current CNN/active-learning/GA notebook uses TensorFlow/Keras, not PyTorch."
python3 - <<'PY'
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy
import sklearn
import joblib
import pygad
print('TensorFlow', tf.__version__)
print('NumPy', np.__version__)
print('pandas', pd.__version__)
print('SciPy', scipy.__version__)
print('scikit-learn', sklearn.__version__)
print('joblib', joblib.__version__)
print('PyGAD', pygad.__version__)
print('TensorFlow devices:', tf.config.list_physical_devices())
PY
