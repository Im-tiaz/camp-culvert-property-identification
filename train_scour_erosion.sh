#!/bin/bash
#SBATCH --job-name=camp_se
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=ad-users
#SBATCH --partition=partition-l
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi || true
export PATH=$HOME/.local/bin:$PATH

cd ~/Code/camp_ml/pipeline
DS=~/Code/culvert_inference/datasets
MODELS=~/Code/camp_ml/models

echo "==================== SCOUR (binary) ===================="
python3 retrain.py --data $DS/scour --task scour --tag field --unfreeze --workers 8
SCOUR=$(ls -t $MODELS/scour_model_field_*.pt | head -1)
echo "Evaluating: $SCOUR"
python3 evaluate_model.py --model "$SCOUR" --data $DS/scour --split test

echo ""
echo "==================== EROSION CONTROL (binary) ===================="
python3 retrain.py --data $DS/erosion_control --task erosion_control --tag field --unfreeze --workers 8
EROS=$(ls -t $MODELS/erosion_control_model_field_*.pt | head -1)
echo "Evaluating: $EROS"
python3 evaluate_model.py --model "$EROS" --data $DS/erosion_control --split test

echo "DONE at $(date)"
