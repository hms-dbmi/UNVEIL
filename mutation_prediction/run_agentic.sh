#!/bin/bash
# Demographic-Aware Agentic Scheduling for Mutation Prediction
# 
# For HPC/SLURM environments: uncomment #SBATCH directives and module load commands
# For local use: run directly with bash run_agentic.sh

#SBATCH -c 4
#SBATCH -t 10:00:00
#SBATCH -p short
#SBATCH --mem=32G
#SBATCH -o ./logs/demographic_agentic/%A_%a.log
#SBATCH -e ./logs/demographic_agentic/%A_%a.log
#SBATCH --array=0-54

cd "$(dirname "${BASH_SOURCE[0]}")"

# HPC environment modules (uncomment if using SLURM)
# module load gcc/14.2.0 cuda/12.8 conda/miniforge3/24.11.3-0
# conda activate unveil

mkdir -p ./logs/demographic_agentic

ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}

# Example configuration
ATTRIBUTE="Age"
CANCER="BRCA"
GENE="PIK3CA"
FOUNDATION_MODEL="CHIEF"

# Validate configuration
if [ -z "$ATTRIBUTE" ] || [ -z "$CANCER" ] || [ -z "$GENE" ] || [ -z "$FOUNDATION_MODEL" ]; then
    echo "ERROR: Missing required variables"
    exit 1
fi

echo "Running: $ATTRIBUTE / $CANCER / $GENE / $FOUNDATION_MODEL"

# Model output path
MODEL_PATH="./output/mutation_models_agentic/${ATTRIBUTE}/TCGA/${FOUNDATION_MODEL}/FS/agent_demographic_agentic/"

# Attribute mapping
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
CANCER_LOWER=$(echo $CANCER | tr '[:upper:]' '[:lower:]')

# Training arguments
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

python main_genetic.py "${CMD_ARGS[@]}"
