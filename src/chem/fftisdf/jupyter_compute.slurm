#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter-%j.out
#SBATCH --error=jupyter-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

PORT=8888

echo "Compute node: $(hostname)"
echo "Port: $PORT"
echo "which python: $(which python)"
python --version
python -m pip -V

echo "Installing/fixing required Jupyter kernel packages..."
python -m pip install --user --upgrade --force-reinstall pygments ipython ipykernel nbconvert

echo "Testing imports..."
python - <<'PY'
import sys
print("Python executable:", sys.executable)

for mod in ["pygments", "IPython", "ipykernel", "nbconvert"]:
    try:
        m = __import__(mod)
        print(mod, "OK", getattr(m, "__version__", ""), getattr(m, "__file__", ""))
    except Exception as e:
        print(mod, "FAILED:", repr(e))
        raise
PY

echo
echo "Starting Jupyter Lab..."
echo

python -m jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT
