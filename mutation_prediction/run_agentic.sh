#!/bin/bash
# ===============================================================================
# HPC-SPECIFIC SCRIPT - Requires SLURM Job Scheduler
# ===============================================================================
# This script is designed for High-Performance Computing (HPC) environments with
# SLURM job scheduler. To run on non-HPC systems:
#
#   1. Remove all #SBATCH directives (lines 8-15)
#   2. Remove or comment out 'module load' command (line 35)
#   3. Manually set ARRAY_ID variable instead of using $SLURM_ARRAY_TASK_ID
#      Example: ARRAY_ID=0
#   4. Create log directories manually: mkdir -p ./logs/demographic_agentic
#   5. Run directly: bash run_agentic.sh
# ===============================================================================
#SBATCH -c 4
#SBATCH -t 10:00:00
#SBATCH -p short
#SBATCH --mem=32G
#SBATCH -o ./logs/demographic_agentic/%A_%a.log
#SBATCH -e ./logs/demographic_agentic/%A_%a.log
#SBATCH --array=0-54

# Demographic-Aware Agentic Scheduling for Mutation Prediction
# Multi-Factor Routing Strategy
#
# Key Innovation:
#   Routes based on THREE factors (zero data leakage):
#   1. Training progress (total slides processed)
#   2. Group imbalance ratio (majority/minority from demographic data)
#   3. Demographic model accuracy (computed from training data)
#
# Decision Logic:
#   - Demo accuracy < 0.60: always random (unreliable model)
#   - Imbalance > 5.0 AND <150 slides: conservative random (preserve minority)
#   - <200 slides processed: conservative random (early training)
#   - >=200 slides processed: signal-leveraging (late training)
#
# Total: 55 jobs (biased tasks across Age/Race/Sex x foundation models)

# Get mutation_prediction directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# Load required modules (modify for your cluster environment)
# Comment out these lines if not running on HPC with environment modules:
# module load gcc/14.2.0 cuda/12.8 conda/miniforge3/24.11.3-0
# conda activate unveil

# Create log directory
mkdir -p ./logs/demographic_agentic

# Set array ID (default to 0 for manual testing)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "WARNING: SLURM_ARRAY_TASK_ID not set. Using 0 for testing."
    ARRAY_ID=0
else
    ARRAY_ID=$SLURM_ARRAY_TASK_ID
fi

# For example/testing purposes, use predefined configuration
# In production, this would use task_config_generator_unified.py
# CONFIG=$(python task_config_generator_unified.py $ARRAY_ID)

# Example configuration for PIK3CA gene
ATTRIBUTE="Age"
CANCER="BRCA"
GENE="PIK3CA"
FOUNDATION_MODEL="CHIEF"

echo ""
echo "=== Job Configuration (Demographic-Aware Agentic Scheduling) ==="
echo "Array ID: $ARRAY_ID"
echo "Attribute: $ATTRIBUTE"
echo "Cancer: $CANCER"
echo "Gene: $GENE"
echo "Foundation Model: $FOUNDATION_MODEL"
echo "Strategy: demographic-aware agentic scheduling (multi-factor routing)"
echo "Innovation: Routes based on imbalance, accuracy, progress"
echo "Factor 1: Demo accuracy (threshold: 0.60)"
echo "Factor 2: Group imbalance (threshold: 5.0)"
echo "Factor 3: Training progress (threshold: 200 slides)"
echo "=================================================================="
echo ""

# Validate all required variables are set
if [ -z "$ATTRIBUTE" ] || [ -z "$CANCER" ] || [ -z "$GENE" ] || [ -z "$FOUNDATION_MODEL" ]; then
    echo "ERROR: One or more required variables are empty"
    echo "  ATTRIBUTE=$ATTRIBUTE"
    echo "  CANCER=$CANCER"
    echo "  GENE=$GENE"
    echo "  FOUNDATION_MODEL=$FOUNDATION_MODEL"
    exit 1
fi

# Build model path
MODEL_PATH="./output/mutation_models_agentic/${ATTRIBUTE}/TCGA/${FOUNDATION_MODEL}/FS/agent_demographic_agentic/"

echo "Model path: $MODEL_PATH"

# Map attribute to sensitive format
case "$ATTRIBUTE" in
    Age)
        SENSITIVE='{"age": ["old", "young"]}'
        ;;
    Race)
        SENSITIVE='{"race": ["White", "Black or African American"]}'
        ;;
    Sex)
        SENSITIVE='{"sex": ["Male", "Female"]}'
        ;;
esac

# Feature length mapping
declare -A LENGTHS=(
    ["CHIEF"]=768
    ["UNI"]=1024
    ["GIGAPATH"]=1536
    ["VIRCHOW2"]=2560
)
INPUT_LENGTH=${LENGTHS[$FOUNDATION_MODEL]}

# Convert cancer to lowercase
CANCER_LOWER=$(echo $CANCER | tr '[:upper:]' '[:lower:]')

# Build command line arguments
CMD_ARGS=(
    --cancer "$CANCER_LOWER"
    --model_path="$MODEL_PATH"
    --embeddings_base_path="./data/features/"
    --partition=0
    --fair_attr="$SENSITIVE"
    --task=4
    --train_method=baseline
    --genes "$GENE"
    --lr=1e-4
    --dropout=0.25
    --seed=0
    --epochs=100
    --n_workers=0
    --batch_size=16
    --eval_batch_size=8
    --acc_grad=2
    --scheduler_step=1
    --scheduler_gamma=0.955
    --device=cpu
    --data_source=TCGA
    --cutoff_method=none
    --input_feature_length=$INPUT_LENGTH
    --foundation_model="$FOUNDATION_MODEL"
    --slide_type=FS
    --magnification=20
    --selection=AUROC
    --max_train_tiles=2000
    --patience=10
    --use_demographic_agent
    --demographic_strategy=unified
    --demographic_base_percentile=15
    --demographic_attribute="$ATTRIBUTE"
    --demographic_use_correctness_weighting
    --demographic_use_v6_routing
)

# Run the main training script
echo "Starting training with demographic-aware agentic scheduling (Multi-Factor Routing)..."
python main_genetic.py "${CMD_ARGS[@]}"

EXIT_CODE=$?
echo "Training completed with exit code: $EXIT_CODE"

exit $EXIT_CODE
