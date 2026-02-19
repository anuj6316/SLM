import urllib.request
from datasets import load_dataset
import os
import json
import yaml
from pprint import pprint
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SPIDER_TABLES_URL = "https://huggingface.co/datasets/xlangai/spider/raw/main/tables.json"

def fetch_spider_tables(output_path="data/raw/spider_tables.json"):
    """
    Downloads the Spider tables.json file if it doesn't exist.
    """
    if os.path.exists(output_path):
        logging.info(f"[*] {output_path} already exists. Skipping download.")
        return output_path
    
    logging.info(f"[*] Downloading Spider tables from {SPIDER_TABLES_URL}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(SPIDER_TABLES_URL, output_path)
        logging.info(f"[+] Downloaded Spider tables to {output_path}")
    except Exception as e:
        logging.error(f"[-] Failed to download Spider tables: {e}")
        return None
    return output_path

def download_and_save(dataset_config: list, output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    local_paths = []

    for ds_info in dataset_config:
        path = ds_info['path']
        split = ds_info['split']

        # format local name
        safe_name = path.replace("/","_")
        target_file = os.path.join(output_dir, f"{safe_name}_{split}.jsonl")
        
        # Specific check for Spider tables
        if "spider" in path.lower():
            fetch_spider_tables(os.path.join(output_dir, "spider_tables.json"))

        if os.path.exists(target_file):
            logging.info(f"[*] {target_file} already exists. Skipping download.")
            local_paths.append(target_file)
            continue

        logging.info(f"[*] Processing: {path} {split}")

        # load from HF
        dataset = load_dataset(path, split=split)

        with open(target_file, "w", encoding="utf-8") as f:
            for entry in tqdm(dataset, desc=f"Writing {safe_name}"):
                f.write(json.dumps(entry) + '\n')

        local_paths.append(target_file)
        logging.info(f"[+] Save to {target_file}")
    return local_paths

def main():
    datasets_list = load_config()
    pprint(cfg)
    pass

if __name__ == "__main__":
    main()