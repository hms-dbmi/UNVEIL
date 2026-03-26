#!/bin/bash

# Demographic Classifier Training Script
# Usage: ./run.sh <config_file>
# Example: ./run.sh configs/example_train.json

cd "$(dirname "${BASH_SOURCE[0]}")"

# Default to example config if no argument provided
CONFIG_FILE="${1:-configs/example_train.json}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Usage: ./run.sh <config_file>"
    echo "Example: ./run.sh configs/example_train.json"
    exit 1
fi

echo "Using config: $CONFIG_FILE"
python run.py --configs_path "$CONFIG_FILE"

