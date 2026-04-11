@echo off
echo === AnimTOON Setup (Windows) ===
echo.

:: Create venv
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Check CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo === Setup Complete ===
echo.
echo Usage:
echo   venv\Scripts\activate.bat
echo   python src/dataset_pipeline.py --limit 1000 --output data/animtoon_train.jsonl
echo   python src/train_animtoon.py --data data/animtoon_train.jsonl
echo   python src/test_inference.py "a bouncing blue ball"
echo.
echo For Unsloth training (faster, less VRAM):
echo   python -m venv venv-unsloth
echo   venv-unsloth\Scripts\activate.bat
echo   pip install unsloth
echo   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --no-deps
echo   pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --no-deps
echo   python src/train_unsloth.py --data data/animtoon_train.jsonl
pause
