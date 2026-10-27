import os, sys
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
from text.g2pw import G2PWPinyin
g2pw = G2PWPinyin(model_dir="GPT_SoVITS/text/G2PWModel",model_source="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",v_to_u=False, neutral_tone_with_five=True)

from huggingface_hub import hf_hub_download, HfApi

REPO_ID = "Geukrak/GPT_zundamon"
api = HfApi()
files = api.list_repo_files(repo_id=REPO_ID)

# Define mappings from file in repo to local directory
downloads = [
    ("zudamon_style_1-e15.ckpt", "GPT_weights_v2"),
    ("zundamon_vits.onnx", "SoVITS_weights_v2"),
]

for f in files:
    if f.startswith("hubert"):
        downloads.append((f, "GPT_SoVITS/pretrained_models/chinese-hubert-base"))
    elif f.startswith("roberta"):
        downloads.append((f, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"))

for file_name, local_dir in downloads:
    os.makedirs(local_dir, exist_ok=True)
    hf_hub_download(repo_id=REPO_ID, filename=file_name, local_dir=local_dir)
    print(f"Downloaded {file_name} to {local_dir}")
