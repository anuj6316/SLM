"""
Publisher Module - Uploads processed datasets to HuggingFace Hub.

Features:
  - Uploads JSONL data and README dataset card
  - Creates versioned tags for releases
  - Includes retry logic with exponential backoff
"""
import os
import time
import logging
from datetime import datetime
from typing import Optional
from functools import wraps

from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("publisher")


def retry_with_backoff(max_retries: int = 3, base_delay: float = 2.0):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles each retry)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
            logger.error(f"All {max_retries + 1} attempts failed")
            raise last_exception
        return wrapper
    return decorator


@retry_with_backoff(max_retries=3, base_delay=2.0)
def _upload_file_with_retry(api: HfApi, token: str, **kwargs) -> None:
    """Uploads a file to HuggingFace Hub with retry logic."""
    api.upload_file(token=token, **kwargs)


@retry_with_backoff(max_retries=3, base_delay=2.0)
def _create_tag_with_retry(api: HfApi, token: str, **kwargs) -> None:
    """Creates a tag on HuggingFace Hub with retry logic."""
    api.create_tag(token=token, **kwargs)


def publish_to_hf(repo_id: str, data_path: str, readme_path: str) -> bool:
    """
    Publishes the dataset and its metadata card to HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "username/dataset-name")
        data_path: Local path to the JSONL data file
        readme_path: Local path to the README.md dataset card

    Returns:
        True if successful, False otherwise
    """
    api = HfApi()
    token = os.getenv("HF_TOKEN")

    if not token:
        logger.error("HF_TOKEN not found. Check your .env file.")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    try:
        logger.info(f"Uploading data to {repo_id}...")
        _upload_file_with_retry(
            api,
            token,
            path_or_fileobj=data_path,
            path_in_repo="train_sft.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Data update {timestamp}"
        )

        if os.path.exists(readme_path):
            logger.info("Uploading Dataset Card (README.md)...")
            _upload_file_with_retry(
                api,
                token,
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Update Dataset Card metadata"
            )

        _create_tag_with_retry(
            api,
            token,
            repo_id=repo_id,
            tag=f"v_{timestamp}",
            repo_type="dataset"
        )
        logger.info(f"Success! Dataset published and tagged as v_{timestamp}")
        return True

    except Exception as e:
        logger.error(f"Failed to publish after retries: {e}")
        return False


if __name__ == "__main__":
    REPO = "anuj6316/text2sql"
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    DATA = os.path.join(BASE_DIR, "data/train_sft.jsonl")
    README = os.path.join(BASE_DIR, "text2sql_dataset_preprocessing/README.md")

    success = publish_to_hf(REPO, DATA, README)
    exit(0 if success else 1)
