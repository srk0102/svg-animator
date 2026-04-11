"""Merge LoRA adapter into base model for continued training."""
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

os.environ.setdefault('HF_HOME', 'D:/hf-cache')

checkpoint = 'models/animtoon-3b-layers/checkpoint-1400'
output = 'models/animtoon-3b-v2-merged'

print(f'Loading base model...')
model = AutoModelForCausalLM.from_pretrained(
    'models/animtoon-3b-merged',
    torch_dtype=torch.float16,
    device_map='cpu'
)
tokenizer = AutoTokenizer.from_pretrained('models/animtoon-3b-merged')

print(f'Loading LoRA from {checkpoint}...')
model = PeftModel.from_pretrained(model, checkpoint)

print('Merging LoRA into base...')
model = model.merge_and_unload()

print(f'Saving to {output}...')
model.save_pretrained(output)
tokenizer.save_pretrained(output)
print('Done! Now train layer-aware LoRA on this merged model.')
