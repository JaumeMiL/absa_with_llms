#!/bin/bash
#SBATCH -p cuda
#SBATCH -A cuda
#SBATCH --qos=cuda3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH -c 2
#SBATCH --mem=48Gb
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch --array=0-N slurm/array.sbatch /artifacts/sweeps/name.jsonl"
  exit 1
fi

cd "$SLURM_SUBMIT_DIR"

RUN_USER="${USER:-$(whoami)}"
if [[ "$RUN_USER" == "mgl1013" ]]; then
  export ABSA_RUN_NAME_SUFFIX="mgl1013"
else
  export ABSA_RUN_NAME_SUFFIX="mgl1008"
fi

source /scratch/nas/1/PDI/mml0/MML.venv/bin/activate
python3 scripts/run_sweep_row.py "$1"
deactivate
