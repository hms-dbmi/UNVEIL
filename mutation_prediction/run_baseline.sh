#!/bin/bash
# ===============================================================================
# HPC-SPECIFIC SCRIPT - Requires SLURM Job Scheduler
# ===============================================================================
# This script is designed for High-Performance Computing (HPC) environments with
# SLURM job scheduler. To run on non-HPC systems:
#
#   1. Remove all #SBATCH directives (lines 8-15)
#   2. Remove or comment out 'module load' command (line 22)
#   3. Manually set IDX variable instead of using $SLURM_ARRAY_TASK_ID
#      Example: IDX=0
#   4. Create log directories manually: mkdir -p ./logs/train_TCGA_mutation
#   5. Run directly: bash run_baseline.sh
# ===============================================================================
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -p short
#SBATCH --mem=16G
#SBATCH -o ./logs/train_TCGA_mutation/%A_%a.log
#SBATCH -e ./logs/train_TCGA_mutation/%A_%a.log
#SBATCH --array=0-30

# Baseline Mutation Prediction for TCGA Cancer Types
# This script runs baseline mutation prediction across 31 TCGA cancer types
# Modify --array range to process specific cancer types

# Get mutation_prediction directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# Load required modules (modify for your cluster environment)
# Comment out these lines if not running on HPC with environment modules:
# module load gcc/14.2.0 cuda/12.8 conda/miniforge3/24.11.3-0
# conda activate unveil

# Create log directory
mkdir -p ./logs/train_TCGA_mutation

# Get array task ID (use 0 for single runs)
IDX=${SLURM_ARRAY_TASK_ID:-0}

# TCGA cancer types (31 total)
declare -a CANCER=("brca" "coadread" "gbm"  "kich" "kirc"   "kirp" "lgg"  "luad" "lusc" "ov" \
                   "ucec" "prad"     "thca" "hnsc" "stad"   "skcm" "blca" "sarc" "lihc" "cesc"  \
                   "paad" "tgct"     "esca" "pcpg" "acc"    "thym" "meso" "ucs"   "uvm" "chol" \
                   "dlbc")

# Foundation model feature dimensions
declare -A INPUT_FEATURE_LENGTHS=(
    ["CHIEF"]=768
    ["UNI"]=1024
    ["GIGAPATH"]=1536
    ["VIRCHOW2"]=2560
)

# Configuration
cancer=${CANCER[$IDX]}
SENSITIVE='{"age": ["old", "young"]}'
SEN="Age"
SLIDE_TYPE="FS"
DATA_SOURCE="TCGA"
FOUNDATION_MODEL="CHIEF"
MAGNIFICATION=20
INPUT_FEATURE_LENGTH=${INPUT_FEATURE_LENGTHS[$FOUNDATION_MODEL]}
INFERENCE_ONLY=false
CUTOFF_METHOD=none
TASK=4  # Mutation prediction task (numeric value)
TRAIN_METHOD="baseline"
EPOCHS=100
N_WORKERS=4
USE_EARLY_STOPPING=true
PATIENCE=10
LIMIT_TILES=true
MAX_TILES=2000
MODEL_PATH="./output/mutation_models/${SEN}/${DATA_SOURCE}/${FOUNDATION_MODEL}/${SLIDE_TYPE}/"
EMBEDDINGS_BASE_PATH="./data/features/"

# Run mutation prediction (BASELINE - no demographic agent)
python main_genetic.py \
    --cancer $cancer \
    --model_path="$MODEL_PATH" \
    --embeddings_base_path="$EMBEDDINGS_BASE_PATH" \
    --partition=0 \
    --fair_attr="$SENSITIVE" \
    --task=$TASK \
    --train_method=$TRAIN_METHOD \
    --lr=1e-4 \
    --dropout=0.25 \
    --seed=0 \
    --epochs=$EPOCHS \
    --n_workers=$N_WORKERS \
    --batch_size=16 \
    --eval_batch_size=8 \
    --acc_grad=2 \
    --scheduler_step=1 \
    --scheduler_gamma=0.955 \
    --device="cpu" \
    --data_source="$DATA_SOURCE" \
    --cutoff_method=$CUTOFF_METHOD \
    --input_feature_length="$INPUT_FEATURE_LENGTH" \
    --foundation_model="$FOUNDATION_MODEL" \
    --slide_type=$SLIDE_TYPE \
    --magnification=$MAGNIFICATION \
    --selection="AUROC" \
    $( [ "$INFERENCE_ONLY" = true ] && echo "--inference_only" ) \
    $( [ "$LIMIT_TILES" = true ] && echo "--max_train_tiles=$MAX_TILES" ) \
    $( [ "$USE_EARLY_STOPPING" = true ] && echo "--patience=$PATIENCE" )
