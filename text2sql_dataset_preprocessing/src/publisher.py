import os
import logging
from datetime import datetime
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("publisher")

def publish_to_hf(repo_id: str, data_path: str, readme_path: str):
    """
    Publishes the dataset and its metadata card to HF Hub.
    """
    api = HfApi()
    token = os.getenv("HF_TOKEN")
    
    if not token:
        logger.error("HF_TOKEN not found. Check your .env file.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    try:
        # 1. Upload the Dataset File
        logger.info(f"Uploading data to {repo_id}...")
        api.upload_file(
            path_or_fileobj=data_path,
            path_in_repo="train_sft.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Data update {timestamp}"
        )

        # 2. Upload the README (The Dataset Card)
        # This fixes the ConfigNamesError by explicitly defining metadata
        if os.path.exists(readme_path):
            logger.info("Uploading Dataset Card (README.md)...")
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message="Update Dataset Card metadata"
            )

        # 3. Create Tag
        api.create_tag(repo_id=repo_id, tag=f"v_{timestamp}", repo_type="dataset", token=token)
        logger.info(f"Success! Dataset published and tagged as v_{timestamp}")
        
    except Exception as e:
        logger.error(f"Failed to publish: {e}")

if __name__ == "__main__":
    REPO = "anuj6316/text2sql"
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    DATA = os.path.join(BASE_DIR, "data/train_sft.jsonl")
    README = os.path.join(BASE_DIR, "text2sql_dataset_preprocessing/README.md")
    
    publish_to_hf(REPO, DATA, README)
