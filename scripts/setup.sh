#!/bin/bash
echo "=== AnimTOON Setup (Mac/Linux) ==="
echo

# Create venv
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check CUDA
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None (CPU mode)')"

echo
echo "=== Setup Complete ==="
echo
echo "Usage:"
echo "  source venv/bin/activate"
echo "  python src/dataset_pipeline.py --limit 1000 --output data/animtoon_train.jsonl"
echo "  python src/train_animtoon.py --data data/animtoon_train.jsonl"
echo "  python src/test_inference.py \"a bouncing blue ball\""
echo
echo "For Unsloth training (faster, less VRAM):"
echo "  python3 -m venv venv-unsloth"
echo "  source venv-unsloth/bin/activate"
echo "  pip install unsloth"
echo "  python src/train_unsloth.py --data data/animtoon_train.jsonl"
