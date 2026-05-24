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

## Use this script to send a job to the cluster with: 
##   sbatch slurm/fewshot.sh
##   sbatch slurm/fewshot.sh configs/experiments/few_shot/single_run.yaml

CONFIG="${1:-configs/experiments/few_shot/single_run.yaml}"

cd "$SLURM_SUBMIT_DIR"

# activate virtual environment with needed python modules
source /scratch/nas/1/PDI/mml0/MML.venv/bin/activate

# run the few-shot experiment config
python3 scripts/run_experiment.py "$CONFIG"

# deactivate the virtual environment
deactivate
