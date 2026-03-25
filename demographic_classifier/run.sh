#!/bin/bash

# Demographic Classifier Training Script
# Usage: ./run.sh <config_file>
# Example: ./run.sh configs/train_example.json

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -z "$1" ]; then
    echo "Usage: ./run.sh <config_file>"
    echo "Example: ./run.sh configs/train_example.json"
    exit 1
fi

python run.py --configs_path "$1"

