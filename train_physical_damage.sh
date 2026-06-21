#!/bin/bash
#SBATCH --job-name=camp_dmg
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=ad-users
#SBATCH --partition=partition-l
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi || true
export PATH=$HOME/.local/bin:$PATH

cd ~/Code/camp_ml/pipeline
DS=~/Code/culvert_inference/datasets
MODELS=~/Code/camp_ml/models

echo "==================== PHYSICAL DAMAGE (4-tier severity) ===================="
python3 retrain.py --data $DS/physical_damage --task physical_damage \
    --tag field --unfreeze --workers 8

MODEL=$(ls -t $MODELS/physical_damage_model_field_*.pt | head -1)
echo "Evaluating: $MODEL"
python3 evaluate_model.py --model "$MODEL" --data $DS/physical_damage --split test

echo "DONE at $(date)"
