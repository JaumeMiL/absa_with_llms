#!/bin/bash
#SBATCH -p cuda
#SBATCH -A cudabig
#SBATCH --qos=cudabig3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH -c 2
#SBATCH --mem=48Gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch slurm/sweep_sequential.sh /artifacts/sweeps/name.jsonl"
  exit 1
fi

cd "$SLURM_SUBMIT_DIR"
source /scratch/nas/1/PDI/mml0/MML.venv/bin/activate

# Automatically count the number of lines in the JSONL file
TOTAL_TASKS=$(wc -l < "$1")
MAX_INDEX=$((TOTAL_TASKS - 1))

echo "Found $TOTAL_TASKS tasks to run (0 to $MAX_INDEX)..."

# Loop through all configurations sequentially
for i in $(seq 0 $MAX_INDEX); do
    echo "=========================================="
    echo "Starting task $i at $(date)"
    echo "=========================================="
    
    # Export the variable so the Python script thinks it's in an array
    export SLURM_ARRAY_TASK_ID=$i
    python3 scripts/run_sweep_row.py "$1"
done

deactivate
echo "All tasks completed successfully!"
