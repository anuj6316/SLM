import subprocess
from huggingface_hub import login, HfApi, snapshot_download
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Setup for the model
repo_id = "google/gemma-3-270m"
download_dir = "./models/gemma-3-270m"

# --- Option 1: Faster & Recommended Way (Recommended) ---
# To use this, just uncomment the block below and ensure 'hf-transfer' is installed.
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# snapshot_download(
#     repo_id=repo_id,
#     local_dir=download_dir,
#     token=hf_token,
#     local_dir_use_symlinks=False,
#     max_workers=8
# )
# print("✅ Download complete via snapshot_download!")

# --- Option 2: aria2c (Current Approach, now fixed with Auth) ---
login(token=hf_token)
api = HfApi()

files = api.list_repo_files(repo_id)
# Filter large useful files
files = [f for f in files if not f.endswith(".md") and not f.startswith(".")]

os.makedirs(download_dir, exist_ok=True)

for file in files:
    url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
    
    cmd = [
        "aria2c",
        "-x", "16",                # 16 connections per file
        "-s", "16",                # 16 parallel segments
        "-k", "1M",                # 1MB chunks
        "--continue=true",         # resume
        "--auto-file-renaming=false",
        f"--header=Authorization: Bearer {hf_token}", # FIXED: Added Auth Header
        "-d", download_dir,
        "-o", file.replace("/", "_"),
        url
    ]
    
    print(f"🚀 Downloading {file}...")
    subprocess.run(cmd)

print("✅ Download tasks finished!")
