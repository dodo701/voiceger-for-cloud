from huggingface_hub import hf_hub_download, HfApi
import os

REPO_ID = "Geukrak/onnx_zundamon"
api = HfApi()
files = api.list_repo_files(repo_id=REPO_ID)

# Define mappings from file in repo to local directory
downloads = [
    ("zundamon_t2s_encoder.onnx", "GPT_weights_v2"),
    ("zundamon_vits.onnx", "SoVITS_weights_v2"),
    ("zundamon_t2s_sdec.onnx", "GPT_weights_v2"),
    ("zundamon_t2s_fsdec.onnx", "GPT_weights_v2"),
    ("zundamon_latent.onnx", "SoVITS_weights_v2"),
    
]

for f in files:
    if f.startswith("hubert"):
        downloads.append((f, "GPT_SoVITS/pretrained_models/chinese-hubert-base"))
    elif f.startswith("roberta"):
        downloads.append((f, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"))


def sanitize_downloaded_filename(file_name):
    name, ext = os.path.splitext(file_name)
    lower_ext = ext.lower()

    if lower_ext in (".safetensors", ".safetensor"):
        return "models" + ext

    if "-" in name:
        name = name.split("-", 1)[1]

    return name + ext


for file_name, local_dir in downloads:
    os.makedirs(local_dir, exist_ok=True)
    downloaded_path = hf_hub_download(repo_id=REPO_ID, filename=file_name, local_dir=local_dir)

    original_name = os.path.basename(downloaded_path)
    renamed_name = sanitize_downloaded_filename(original_name)
    target_path = os.path.join(os.path.dirname(downloaded_path), renamed_name)

    if original_name != renamed_name:
        os.rename(downloaded_path, target_path)
        print(f"Renamed {original_name} -> {renamed_name}")

    print(f"Downloaded {renamed_name} to {os.path.dirname(target_path)}")
