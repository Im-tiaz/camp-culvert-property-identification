#!/bin/bash
#SBATCH --job-name=camp_es4
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

echo "==================== END SECTION (4-class, CBC dropped) ===================="
python3 retrain.py --data $DS/end_section_4class --task end_section_4class \
    --tag field --unfreeze --workers 8

MODEL=$(ls -t $MODELS/end_section_4class_model_field_*.pt | head -1)
echo "Evaluating: $MODEL"
python3 evaluate_model.py --model "$MODEL" --data $DS/end_section_4class --split test

echo "DONE at $(date)"