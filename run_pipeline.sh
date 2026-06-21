#!/bin/bash
#SBATCH --job-name=camp_infer
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=ad-users
#SBATCH --partition=partition-l
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi || true
export PATH=$HOME/.local/bin:$PATH

cd ~/Code/camp_ml/pipeline
OUT=~/Code/camp_ml/outputs

# 1. ML inference (all 5 models) with tuned thresholds
python3 run_inference.py --images ~/Code/all_images --per-image
#python3 run_inference.py --images ~/Code/all_images --per-image --shape-conf 0.30

# 2. rule-based dimension check
python3 dimension_check.py --table ~/Code/nmdot_table_fixed.csv

# 3. merge everything on Culvert ID (newest files auto-picked)
PRED=$(ls -t $OUT/camp_predictions_v*.csv | head -1)
DIMS=$(ls -t $OUT/dimension_check_*.csv | head -1)
echo "Merging: $PRED + $DIMS"
python3 merge_results.py --table ~/Code/nmdot_table_fixed.csv \
    --predictions "$PRED" --dimensions "$DIMS"

echo "DONE at $(date)"