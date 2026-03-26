#!/bin/bash
# Demographic Classifier Training
# Usage: ./run.sh [config_file]
# Default: configs/example_train.json

cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG_FILE="${1:-configs/example_train.json}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

python run.py --configs_path "$CONFIG_FILE"

