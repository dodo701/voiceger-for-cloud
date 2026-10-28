from huggingface_hub import hf_hub_download, HfApi
import os

REPO_ID = "Geukrak/GPT_zundamon"
api = HfApi()
files = api.list_repo_files(repo_id=REPO_ID)

# Define mappings from file in repo to local directory
downloads = [
    ("zudamon_style_1-e15.ckpt", "GPT_weights_v2"),
    ("zundamon_vits.onnx", "SoVITS_weights_v2"),
    ("zudamon_style_1_e8_s96.pth", "SoVITS_weights_v2") # temporary, for compatability issues.
    
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
