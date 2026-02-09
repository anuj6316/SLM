import json
import os
from tqdm import tqdm

SPIDER_DIR = "data/spider"
os.makedirs(SPIDER_DIR, exist_ok=True)
OUTPUT_DIR = "data/spider_sft"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a question, "
    "generate a correct SQL query. Return only SQL without explanation."
)


# --------------------------------------------------
# Load schemas from tables.json
# --------------------------------------------------
def load_schemas():
    path = os.path.join(SPIDER_DIR, "tables.json")
    with open(path, "r", encoding="utf-8") as f:
        tables = json.load(f)

    schemas = {}

    for db in tables:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        column_names = db["column_names_original"]

        schema_dict = {}

        for table_id, column_name in column_names:
            if table_id == -1:
                continue
            table = table_names[table_id]
            schema_dict.setdefault(table, []).append(column_name)

        # Format schema text
        schema_lines = []
        for table, cols in schema_dict.items():
            schema_lines.append(f"{table}({', '.join(cols)})")

        schemas[db_id] = "\n".join(schema_lines)

    return schemas


# --------------------------------------------------
# Convert single example
# --------------------------------------------------
def convert_example(example, schemas):
    db_id = example["db_id"]
    schema_text = schemas[db_id]

    user_prompt = (
        f"Database Schema:\n{schema_text}\n\n"
        f"Question: {example['question']}"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example["query"]}
        ],
        "metadata": {
            "db_id": db_id
        }
    }


# --------------------------------------------------
# Convert dataset split
# --------------------------------------------------
def process_split(input_file, output_file, schemas):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_file, "w", encoding="utf-8") as out:
        for ex in tqdm(data):
            item = convert_example(ex, schemas)
            out.write(json.dumps(item, ensure_ascii=False) + "\n")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Loading schemas...")
    schemas = load_schemas()
    print(f"Schemas loaded: {len(schemas)}")

    train_file = os.path.join(SPIDER_DIR, "train_spider.json")
    dev_file = os.path.join(SPIDER_DIR, "dev.json")

    train_out = os.path.join(OUTPUT_DIR, "train.jsonl")
    dev_out = os.path.join(OUTPUT_DIR, "validation.jsonl")

    print("Processing train...")
    process_split(train_file, train_out, schemas)

    print("Processing validation...")
    process_split(dev_file, dev_out, schemas)

    print("\nDone")
    print(f"Train saved to: {train_out}")
    print(f"Validation saved to: {dev_out}")


if __name__ == "__main__":
    main()
