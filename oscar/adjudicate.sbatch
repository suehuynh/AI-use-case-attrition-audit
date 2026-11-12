#!/bin/bash
# Pairwise LLM adjudication.
# Usage:
#   sbatch slurm/adjudicate.sbatch                                         # pilot, default model
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct slurm/adjudicate.sbatch
#   sbatch --export=ALL,MODE=full slurm/adjudicate.sbatch                  # all 8,260 pairs
#
# Edit the partition, account, and module lines for your cluster (check its docs).

#SBATCH --job-name=adjudicate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
MODE=${MODE:-pilot}
BACKEND=${BACKEND:-vllm}

# module load cuda
module load python/3.11.11-5e66
source ~/envs/attrition/bin/activate

# Keep model weights in scratch
export HF_HOME=${HF_HOME:-$HOME/scratch/hf_cache}

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
nvidia-smi

PILOT_FLAG=""
[ "$MODE" = "pilot" ] && PILOT_FLAG="--pilot"

python scripts/run_adjudication.py --model "$MODEL" --backend "$BACKEND" $PILOT_FLAG
