import json
import argparse
import os
from schema_utils import (
    load_schema_dict,
    find_relevant_tables,
    expand_with_foreign_keys,
    format_schema,
)


def build_jsonl(spider_dir, output_path):
    tables_path = os.path.join(spider_dir, "tables.json")
    train_path = os.path.join(spider_dir, "train_spider.json")

    print(f"Loading schemas from: {tables_path}")
    schemas = load_schema_dict(tables_path)

    print(f"Loading training data from: {train_path}")
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        print(f"Processing {len(data)} examples...")
        for ex in data:
            db_id = ex["db_id"]
            question = ex["question"]
            sql = ex["query"]

            schema = schemas[db_id]

            # Step 1: table matching
            matched = find_relevant_tables(question, schema)

            # Step 2: FK expansion (join safety)
            selected_tables = expand_with_foreign_keys(matched, schema)

            # Step 3: format schema
            schema_text = format_schema(db_id, schema, selected_tables)

            # Step 4: build instruction
            prompt = f"""You are a Text-to-SQL expert.

{schema_text}

Question: {question}
SQL:"""

            row = {
                "instruction": prompt,
                "output": sql,
            }

            out.write(json.dumps(row) + "\n")

    print("Saved:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider_dir", type=str, default="data/spider", help="Directory containing Spider dataset (tables.json, train_spider.json)")
    parser.add_argument("--output_path", type=str, default="data/spider_sft/train_sft.jsonl", help="Output JSONL file path")
    
    args = parser.parse_args()
    
    build_jsonl(args.spider_dir, args.output_path)
