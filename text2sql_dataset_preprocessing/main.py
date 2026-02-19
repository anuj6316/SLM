import os
import json
import yaml
import logging
from tqdm import tqdm
from pprint import pprint
from src.gatherer import download_and_save
from src.cleaner import clean_and_validate
from src.formatter import normalize_to_sft
from src.schema_parser import load_spider_schemas
from src.schema_generator import generate_schema_map # New Import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_dataset_type(filename: str) -> str:
    if "spider" in filename.lower(): return "spider"
    if "bird" in filename.lower(): return "bird"
    if "gretel" in filename.lower(): return "gretel"
    return "unknown"

def process_datasets(raw_files: list, output_file: str, spider_tables_path: str):
    # 1. Try loading external schemas
    schema_map = load_spider_schemas(spider_tables_path)
    
    # 2. Fallback: If schemas are empty, generate them from the data itself
    if not schema_map:
        logging.info("External schemas not found or invalid. Triggering Reverse Engineering from raw data...")
        schema_map = generate_schema_map(raw_files)
        logging.info(f"Generated {len(schema_map)} database schemas from SQL analysis.")

    processed_count = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in raw_files:
            if not os.path.exists(file_path):
                continue
                
            filename = os.path.basename(file_path)
            ds_type = get_dataset_type(filename)
            logging.info(f"--- Processing {filename} as {ds_type} ---")

            with open(file_path, 'r', encoding='utf-8') as in_f:
                for line in tqdm(in_f, desc=f"Processing {ds_type}"):
                    try:
                        entry = json.loads(line)
                        cleaned = clean_and_validate(entry, ds_type)
                        if not cleaned: continue

                        formatted = normalize_to_sft(cleaned, ds_type, schema_map)
                        if formatted:
                            out_f.write(json.dumps(formatted) + '\n')
                            processed_count += 1
                            
                    except Exception as e:
                        logging.error(f"Error processing entry: {e}")

    logging.info(f"Success: Processed {processed_count} SFT entries to {output_file}.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.yml')
    cfg = load_config(config_path)
    pprint(cfg)
    raw_dir = os.path.join(base_dir, "..", "data", "raw")
    raw_files = download_and_save(cfg['datasets'], output_dir=raw_dir)

    spider_tables = os.path.join(raw_dir, "spider_tables.json")
    output_sft = os.path.join(base_dir, "..", "data", "train_sft.jsonl")
    
    process_datasets(raw_files, output_sft, spider_tables)
