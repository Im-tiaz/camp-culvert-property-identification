#!/bin/bash
#SBATCH --job-name=camp_train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=ad-users
#SBATCH --partition=partition-l
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi || true

export PATH=$HOME/.local/bin:$PATH

cd ~/Code/camp_ml/pipeline
DS=~/Code/culvert_inference/datasets
MODELS=~/Code/camp_ml/models

run_and_eval () {
    local name="$1" data="$2" task="$3" tag="$4"
    echo ""
    echo "==================== ${name} ===================="
    echo "[train] task=${task} tag=${tag}  $(date)"
    python3 retrain.py --data "${data}" --task "${task}" --tag "${tag}" --unfreeze --workers 8

    local model
    model=$(ls -t ${MODELS}/${task}_model_${tag}_*.pt | head -1)
    echo "[eval] model=${model}"
    python3 evaluate_model.py --model "${model}" --data "${data}" --split test
}

run_and_eval "1. END SECTION (5-class)"      "$DS/end_section_combined" end_section    field
run_and_eval "2. SILTING BINARY (unfrozen)"  "$DS/silting_binary"       silting_binary unfrozen
run_and_eval "3. CORROSION (unfrozen)"       "$DS/corrosion_field"      corrosion      unfrozen

echo ""
echo "==================== ALL DONE at $(date) ===================="
