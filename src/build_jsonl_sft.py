import json
import argparse
import os
from tqdm import tqdm
from src.schema_utils import load_schema_dict, build_sft_prompt
from src.utils import get_logger

logger = get_logger(__name__)

def build_jsonl(spider_dir, output_path):
    tables_path = os.path.join(spider_dir, "tables.json")
    train_path = os.path.join(spider_dir, "train_spider.json")

    logger.info(f"Loading schemas from: {tables_path}")
    schemas = load_schema_dict(tables_path)

    logger.info(f"Loading training data from: {train_path}")
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        logger.info(f"Processing {len(data)} examples...")
        for ex in tqdm(data, desc="Building JSONL"):
            db_id = ex["db_id"]
            question = ex["question"]
            sql = ex["query"]

            schema = schemas[db_id]

            # Use centralized prompt builder for consistency
            prompt = build_sft_prompt(db_id, schema, question)

            row = {
                "instruction": prompt,
                "output": sql,
            }

            out.write(json.dumps(row) + "\n")

    logger.info(f"Dataset preparation complete. Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider_dir", type=str, default="data/spider", help="Directory containing Spider dataset")
    parser.add_argument("--output_path", type=str, default="data/spider_sft/train_sft.jsonl", help="Output JSONL file path")
    
    args = parser.parse_args()
    
    build_jsonl(args.spider_dir, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider_dir", type=str, default="data/spider", help="Directory containing Spider dataset (tables.json, train_spider.json)")
    parser.add_argument("--output_path", type=str, default="data/spider_sft/train_sft.jsonl", help="Output JSONL file path")
    
    args = parser.parse_args()
    
    build_jsonl(args.spider_dir, args.output_path)
