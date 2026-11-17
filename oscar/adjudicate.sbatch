#!/bin/bash
# Pairwise LLM adjudication on one GPU (Oscar).
# Usage (from the repo root on Oscar):
#   sbatch --export=ALL,STAGE=pilot,MODEL=Qwen/Qwen2.5-7B-Instruct oscar/adjudicate.sbatch
#   sbatch --time=06:00:00 --export=ALL,STAGE=full,MODEL=<chosen model> oscar/adjudicate.sbatch
# Pilot judges exactly outputs/pilot/pilot_pairs.csv; full judges all retrieval candidates.
# Writes outputs/adjudication/<stage>__<model>.csv and .json

#SBATCH --job-name=adjudicate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
STAGE=${STAGE:-pilot}
MODEL=${MODEL:?set MODEL, e.g. --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct}
BACKEND=${BACKEND:-vllm}

module load python/3.11.11-5e66
source ~/envs/attrition/bin/activate   # venv built from requirements-gpu.txt

# Keep model weights in scratch, not the home quota.
# Llama and Gemma are gated: run `huggingface-cli login` once on a login node first.
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/scratch/hf_cache}
export VLLM_USE_FLASHINFER_SAMPLER=0

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
nvidia-smi

python scripts/run_adjudication.py --stage "$STAGE" --model "$MODEL" --backend "$BACKEND"
