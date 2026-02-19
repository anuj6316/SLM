import json
import argparse
import os
import logging
from typing import Dict, Any, List
from tqdm import tqdm
from schema_utils import load_schema_dict, build_sft_prompt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def build_jsonl(spider_dir: str, output_path: str):
    """
    Processes the Spider dataset and generates a JSONL file suitable for SFT.
    Includes database schema injection into the instructions.
    """
    tables_path = os.path.join(spider_dir, "tables.json")
    train_path = os.path.join(spider_dir, "train_spider.json")

    # 1. Load resources
    try:
        schemas = load_schema_dict(tables_path)
        logger.info(f"Successfully loaded {len(schemas)} database schemas.")
        
        with open(train_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} training examples.")
    except Exception as e:
        logger.critical(f"Initialization failed: {str(e)}")
        return

    # 2. Process and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Starting SFT dataset generation: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as out:
        for ex in tqdm(data, desc="Processing Examples"):
            db_id = ex["db_id"]
            question = ex["question"]
            sql = ex["query"]

            if db_id not in schemas:
                logger.warning(f"Schema for {db_id} not found. Skipping example.")
                continue

            schema = schemas[db_id]

            # Use the centralized prompt builder from schema_utils
            prompt = build_sft_prompt(db_id, schema, question)

            row = {
                "instruction": prompt,
                "output": sql,
            }

            out.write(json.dumps(row) + "\n")

    logger.info(f"Successfully saved {len(data)} examples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spider to SFT JSONL Converter")
    parser.add_argument(
        "--spider_dir", 
        type=str, 
        default="data/spider", 
        help="Directory containing Spider dataset (tables.json, train_spider.json)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="data/spider_sft/train_sft.jsonl", 
        help="Output JSONL file path"
    )
    
    args = parser.parse_args()
    
    try:
        build_jsonl(args.spider_dir, args.output_path)
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")