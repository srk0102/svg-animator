# Kaggle Training Guide — AnimTOON v3.1 / v4.1

End-to-end steps to train both models on Kaggle's free GPUs.

## One-time setup

### 1. Build the training JSONL locally
```powershell
python src/build_training_jsonl.py
```
Outputs `data/iconscout_training_v4.jsonl` (~40-80 MB, ~84k training pairs).

### 2. Upload training data to Kaggle as a private dataset
1. Go to https://www.kaggle.com/datasets → **New Dataset**
2. Upload `data/iconscout_training_v4.jsonl`
3. Title: **animtoon-iconscout-v4**
4. Visibility: **Private**
5. Click **Create**

### 3. Add your HuggingFace token as a Kaggle Secret
1. Kaggle → Notebooks → your notebook → **Add-ons → Secrets**
2. Add secret with:
   - Label: `HF_TOKEN`
   - Value: your HF token (must have access to `srk0102200/AnimTOON-3B-v4` private repo)

## Training run

### 4. Upload the notebook
1. Kaggle → **Create → New Notebook**
2. **File → Import Notebook → Upload** `kaggle/train_animtoon.ipynb`
3. Settings (right sidebar):
   - **Accelerator**: `GPU T4 x2` (or `GPU P100` if T4 unavailable)
   - **Internet**: On
   - **Persistence**: Variables and files
4. Add data: **+ Add Data → search `animtoon-iconscout-v4` → Add**
5. Pin `HF_TOKEN` secret

### 5. Run v4.1 first (the priority)
1. Open Cell 2 → set `BASE_MODEL = "v4"`
2. **Run all** cells
3. Wait ~2-3 hours
4. At the end, download the zip from the Output sidebar
5. Name it `animtoon-3b-v41-adapter.zip`

### 6. Run v3.1 for comparison
1. Same notebook → Cell 2 → change to `BASE_MODEL = "v3"`
2. **Run all** again (Kaggle resets state so you start fresh)
3. Wait another ~2-3 hours
4. Download `animtoon-3b-v31-adapter.zip`

## After training

### 7. Local comparison
```powershell
# Unzip both adapters into models/
7z x animtoon-3b-v31-adapter.zip -omodels\animtoon-3b-v31-adapter
7z x animtoon-3b-v41-adapter.zip -omodels\animtoon-3b-v41-adapter

# Run benchmark (needs compare script updated to load LoRA adapters)
python src/compare_v3_v4.py --v3 models/animtoon-3b-v31-adapter --v4 models/animtoon-3b-v41-adapter
```

### 8. Pick the winner
- Inspect `data/compare_v3_v4/*.animtoon.txt` outputs
- Render `.lottie` files using `test_inference.py` for visual check
- Pick whichever produces better animations

### 9. Ship the winner
- Merge LoRA into base: `python src/merge_lora.py --adapter models/animtoon-3b-XX-adapter --base Qwen/Qwen2.5-3B-Instruct`
- Push to HF: replaces current `srk0102200/AnimTOON-3B` (make it public)
- Update README + model card

## Troubleshooting

### "HF_TOKEN not found"
- Check Kaggle **Add-ons → Secrets** has `HF_TOKEN` defined
- Secret must be **pinned** to the notebook (click pin icon)

### "Cannot access private model srk0102200/AnimTOON-3B-v4"
- Your HF token needs read access to the private repo
- Regenerate token at https://huggingface.co/settings/tokens with "Read" scope

### "CUDA OOM"
- Drop `BATCH_SIZE` from 2 to 1
- Increase `GRAD_ACCUM` from 8 to 16 (keeps effective batch = 16)

### "Dataset not found at /kaggle/input/..."
- Make sure you clicked **+ Add Data** and actually attached `animtoon-iconscout-v4`
- The notebook auto-searches for any `.jsonl` under `/kaggle/input` as a fallback

### Training is too slow
- Make sure Accelerator is `GPU T4 x2` or `GPU P100`, not CPU
- Check you haven't hit the 9h session limit
