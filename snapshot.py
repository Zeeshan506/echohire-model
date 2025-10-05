from huggingface_hub import snapshot_download

repo_id = "Zeeshan506/echohire-qgen-distilgpt2"
local_dir = snapshot_download(repo_id)

print("Model downloaded to:", local_dir)