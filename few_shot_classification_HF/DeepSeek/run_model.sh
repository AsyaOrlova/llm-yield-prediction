#!/bin/sh
#export PATH=$HOME/.local/bin:$PATH
#export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH
#export OLLAMA_MODELS="/mnt/tank/scratch/bgrechkin/.ollama/models"
#sleep 60
# rm -rf ../.ollama/ # clear ollama model and cache (VERY CAREFUL!)
python vsegpt_api.py

# sbatch --cpus-per-task=10 -p aichem --gres=gpu:1 --mem=20G --error=Logs/job_%j.err --output=Logs/job_%j.out run_model.sh