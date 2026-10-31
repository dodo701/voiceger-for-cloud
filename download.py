from huggingface_hub import hf_hub_download
import os

REPO_ID = "Geukrak/onnx_zundamon"

# Define mappings from file in repo to local directory
downloads = [
    ("zudamon_style_1-e15.safetensors", "GPT_weights_v2"),
    ("zundamon_vits.onnx", "SoVITS_weights_v2"),
]

for file_name, local_dir in downloads:
    os.makedirs(local_dir, exist_ok=True)
    downloaded_path = hf_hub_download(repo_id=REPO_ID, filename=file_name, local_dir=local_dir)

    original_name = os.path.basename(downloaded_path)

    print(f"Downloaded {original_name} to {os.path.dirname(downloaded_path)}")
