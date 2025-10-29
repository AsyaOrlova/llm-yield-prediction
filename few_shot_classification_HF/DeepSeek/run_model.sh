#!/bin/sh
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH
rm -rf ~/.ollama/ # clear ollama model and cache (VERY CAREFUL!)
python main.py Data/USPTO_R_text.csv gpt-oss:20b

# sbatch --cpus-per-task=10 -p aichem --gres=gpu:1 --mem=20G --error=Logs/job_%j.err --output=Logs/job_%j.out run_model.sh