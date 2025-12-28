#!/bin/bash
set -e

ENV_NAME="multimodal-hcc"
YML_FILE="environment.yml"

echo "Creating conda environment '$ENV_NAME' from $YML_FILE..."

if ! command -v conda &> /dev/null; then
    echo "Error: conda is not found in PATH."
    exit 1
fi

conda env create -f $YML_FILE

echo ""
echo "Environment '$ENV_NAME' created successfully."
echo "To activate it, run:"
echo "    conda activate $ENV_NAME"
