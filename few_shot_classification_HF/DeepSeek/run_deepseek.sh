#!/bin/sh
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH
python main.py Data/USPTO_R_text.csv deepseek-r1:8b-0528-qwen3-q8_0

# sbatch --cpus-per-task=10 -p aichem --gres=gpu:1 --mem=20G --error=Logs/job_%j.err --output=Logs/job_%j.out run_deepseek.sh