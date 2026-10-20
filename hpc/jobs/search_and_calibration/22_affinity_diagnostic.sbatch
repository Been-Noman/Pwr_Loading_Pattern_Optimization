#!/bin/bash
#SBATCH --job-name=affdiag48
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:05:00
#SBATCH --output=run/logs/affdiag48_%j.out
#SBATCH --error=run/logs/affdiag48_%j.err

set -euo pipefail

echo "===== BATCH SHELL ====="
python - <<'PY'
import os
print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))
print("sched_getaffinity   =", sorted(os.sched_getaffinity(0)))
print("affinity_count      =", len(os.sched_getaffinity(0)))
PY

echo
echo "===== SRUN CPU-BIND=NONE ====="
srun -n 1 -c 48 --cpu-bind=none python - <<'PY'
import os
print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))
print("sched_getaffinity   =", sorted(os.sched_getaffinity(0)))
print("affinity_count      =", len(os.sched_getaffinity(0)))
PY

echo
echo "===== SRUN CPU-BIND=CORES ====="
srun -n 1 -c 48 --cpu-bind=cores python - <<'PY'
import os
print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))
print("sched_getaffinity   =", sorted(os.sched_getaffinity(0)))
print("affinity_count      =", len(os.sched_getaffinity(0)))
PY
