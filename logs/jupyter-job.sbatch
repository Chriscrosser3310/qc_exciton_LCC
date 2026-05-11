#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=/home/jchen9/qc_exciton_LCC/logs/jupyter-%j.out
#SBATCH --error=/home/jchen9/qc_exciton_LCC/logs/jupyter-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd "/home/jchen9/qc_exciton_LCC"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

PORT=8888

echo "Compute node: $(hostname)"
echo "Port: $PORT"
echo "Workdir: $(pwd)"
echo "Python: $(which python)"
python --version

echo "Checking Jupyter dependencies..."
python - <<'PY'
for mod in ["pygments", "IPython", "ipykernel", "jupyterlab"]:
    m = __import__(mod)
    print(mod, "OK", getattr(m, "__version__", ""))
PY

echo
echo "Starting Jupyter Lab..."
echo

python -m jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT
