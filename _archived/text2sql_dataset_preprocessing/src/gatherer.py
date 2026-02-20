"""
Dataset Gatherer Module - Downloads Text-to-SQL datasets from HuggingFace.

Supports: Spider, BIRD, Gretel synthetic datasets.
"""
import urllib.request
import hashlib
from datasets import load_dataset
import os
import json
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger("gatherer")

SPIDER_TABLES_URL = "https://huggingface.co/datasets/xlangai/spider/raw/main/tables.json"
SPIDER_TABLES_SHA256 = None  # TODO: Add hash for integrity verification


def fetch_spider_tables(output_path: str = "data/raw/spider_tables.json") -> Optional[str]:
    """
    Downloads the Spider tables.json file if it doesn't exist.

    Args:
        output_path: Local path to save tables.json

    Returns:
        Path to downloaded file, or None on failure
    """
    if os.path.exists(output_path):
        logger.info(f"[*] {output_path} already exists. Skipping download.")
        return output_path

    logger.info(f"[*] Downloading Spider tables from {SPIDER_TABLES_URL}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(SPIDER_TABLES_URL, output_path)
        logger.info(f"[+] Downloaded Spider tables to {output_path}")
    except Exception as e:
        logger.error(f"[-] Failed to download Spider tables: {e}")
        return None
    return output_path


def download_and_save(
    dataset_config: List[Dict[str, str]],
    output_dir: str = "data/raw"
) -> List[str]:
    """
    Downloads datasets from HuggingFace and saves as JSONL files.

    Args:
        dataset_config: List of dicts with 'path' and 'split' keys
        output_dir: Directory to save downloaded files

    Returns:
        List of local file paths to downloaded JSONL files
    """
    os.makedirs(output_dir, exist_ok=True)
    local_paths = []

    for ds_info in dataset_config:
        path = ds_info['path']
        split = ds_info['split']

        safe_name = path.replace("/", "_")
        target_file = os.path.join(output_dir, f"{safe_name}_{split}.jsonl")

        if "spider" in path.lower():
            fetch_spider_tables(os.path.join(output_dir, "spider_tables.json"))

        if os.path.exists(target_file):
            logger.info(f"[*] {target_file} already exists. Skipping download.")
            local_paths.append(target_file)
            continue

        logger.info(f"[*] Processing: {path} {split}")

        dataset = load_dataset(path, split=split)

        with open(target_file, "w", encoding="utf-8") as f:
            for entry in tqdm(dataset, desc=f"Writing {safe_name}"):
                f.write(json.dumps(entry) + '\n')

        local_paths.append(target_file)
        logger.info(f"[+] Saved to {target_file}")

    return local_paths
