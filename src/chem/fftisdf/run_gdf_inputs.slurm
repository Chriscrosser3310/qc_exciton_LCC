#!/bin/bash
#SBATCH --job-name=gdf-inputs
#SBATCH --output=gdf-%A_%a.out
#SBATCH --error=gdf-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Workdir: $(pwd)"
echo "Python: $(which python)"
python --version

mkdir -p data_GDF

for NK in "$@"; do
    echo "Starting nk=$NK"
    python -u do_scf_gdf.py "$NK"
    echo "Finished nk=$NK"
done
