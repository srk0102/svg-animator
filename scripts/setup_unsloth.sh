#!/bin/bash
echo "=== AnimTOON Unsloth Setup (Mac/Linux) ==="
echo

# Create separate venv for Unsloth
echo "Creating Unsloth virtual environment..."
python3 -m venv venv-unsloth
source venv-unsloth/bin/activate

# Install Unsloth
echo "Installing Unsloth..."
pip install unsloth

# Verify
echo
echo "Verifying installation..."
python3 -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); from unsloth import FastLanguageModel; print('Unsloth OK')"

echo
echo "=== Unsloth Setup Complete ==="
echo
echo "Usage:"
echo "  source venv-unsloth/bin/activate"
echo "  python src/train_unsloth.py --data data/animtoon_train.jsonl --lora-dropout 0"
