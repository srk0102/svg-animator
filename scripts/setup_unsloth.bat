@echo off
echo === AnimTOON Unsloth Setup (Windows) ===
echo.

:: Create separate venv for Unsloth
echo Creating Unsloth virtual environment...
python -m venv venv-unsloth
call venv-unsloth\Scripts\activate.bat

:: Install Unsloth
echo Installing Unsloth...
pip install unsloth

:: Install CUDA torch nightly (required for RTX 50-series / Blackwell GPUs)
echo Installing PyTorch CUDA nightly...
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --no-deps
pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --no-deps

:: Verify
echo.
echo Verifying installation...
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); from unsloth import FastLanguageModel; print('Unsloth OK')"

echo.
echo === Unsloth Setup Complete ===
echo.
echo Usage:
echo   venv-unsloth\Scripts\activate.bat
echo   python src/train_unsloth.py --data data/animtoon_train.jsonl --lora-dropout 0
pause
