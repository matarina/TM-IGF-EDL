#!/bin/bash
set -e
source /deeplearn/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-hcc

echo "Uninstalling conda torch packages..."
# Force uninstall to clear the way
conda remove -y --force pytorch torchvision torchaudio pytorch-cuda || true
pip uninstall -y torch torchvision torchaudio || true

echo "Installing PyTorch via PIP (CUDA 11.8)..."
# Using the official pytorch wheel for cu118
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

echo "Reinstalling PyG dependencies..."
pip install --force-reinstall \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.1+cu118.html

echo "Verifying import..."
python -c "import torch; import torch_scatter; print('Success! Torch version:', torch.__version__)"
