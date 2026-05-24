#! /bin/bash
#SBATCH -p cuda
#SBATCH -A cuda
#SBATCH --qos=cuda3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH -c 2
#SBATCH --mem=48Gb 
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch slurm/single_gpu.sbatch configs/experiments/zero_shot/single_llama.yaml"
  exit 1
fi

cd "$SLURM_SUBMIT_DIR"

source /scratch/nas/1/PDI/mml0/MML.venv/bin/activate
python3 scripts/run_experiment.py "$1"
deactivate
