#!/usr/bin/env python3
from __future__ import annotations
import json, os, platform, sys, time
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas, scipy, sklearn, joblib, psutil, pygad
from cnn_architecture import configure_tensorflow, create_model, set_tensorflow_seed

threads, inter = configure_tensorflow()
set_tensorflow_seed(42)
model = create_model(train_size=64, batch_size=32)
x = np.zeros((64, 15, 15, 3), dtype=np.float32)
# Small valid-shaped synthetic tensor for a forward/backward software test only.
rng = np.random.default_rng(42)
for i in range(64):
    x[i] = rng.random((15, 15, 3), dtype=np.float32)
y = np.zeros((64, 1), dtype=np.float32)
start = time.perf_counter()
h = model.fit(x, {'keff': y, 'fa': y, 'fqr': y}, epochs=1, batch_size=32, verbose=0)
pred = model.predict(x[:2], verbose=0)
report = {
    'python': sys.version.split()[0],
    'platform': platform.platform(),
    'hostname': platform.node(),
    'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
    'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
    'tensorflow': tf.__version__,
    'numpy': np.__version__,
    'pandas': pandas.__version__,
    'scipy': scipy.__version__,
    'scikit_learn': sklearn.__version__,
    'joblib': joblib.__version__,
    'psutil': psutil.__version__,
    'pygad': pygad.__version__,
    'tf_devices': [str(d) for d in tf.config.list_physical_devices()],
    'threads': threads,
    'inter_threads': inter,
    'one_epoch_loss': float(h.history['loss'][-1]),
    'prediction_shapes': [list(p.shape) for p in pred],
    'elapsed_seconds': time.perf_counter() - start,
    'status': 'PASS',
}
Path('ml_preflight_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
