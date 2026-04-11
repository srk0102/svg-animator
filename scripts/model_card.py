"""Upload model card to HuggingFace."""
from huggingface_hub import HfApi
import os

os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')
api = HfApi(token=os.environ['HF_TOKEN'])

with open('MODEL_CARD.md', 'r', encoding='utf-8') as f:
    content = f.read()

api.upload_file(
    path_or_fileobj=content.encode('utf-8'),
    path_in_repo='README.md',
    repo_id='srk0102200/AnimTOON-3B',
    commit_message='Update model card with full documentation',
)
print('Model card uploaded!')
