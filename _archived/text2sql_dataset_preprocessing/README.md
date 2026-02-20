# Text2SQL Preprocessing Module

![Text-to-SQL MLOps Pipeline Flow](https://storage.googleapis.com/second-petal-295822.appspot.com/elements/elements%3Ad6a8a796d7c03b962ceaf006140dbc0218daf65f7ac9c3b5595641b2c787654b.png)

This module transforms raw Text-to-SQL datasets into high-quality training artifacts for SLMs.

## 🔄 Core Workflow
1. **Gather (`src/gatherer.py`):** Automatically downloads Spider, BIRD, and Gretel datasets from Hugging Face.
2. **Generate Schema (`src/schema_generator.py`):** If external metadata is missing, this component reverse-engineers database schemas by parsing all available SQL queries.
3. **Clean (`src/cleaner.py`):** Uses `sqlglot` to normalize SQL and sanitize question text.
4. **Format (`src/formatter.py`):** Wraps data into the "Instruction-Input-Output" pattern.
5. **Publish (`src/publisher.py`):** Pushes the final JSONL and its Dataset Card to Hugging Face Hub with version tagging.

## 🛠 Usage
Run via Poe the Poet:
```bash
uv run poe preprocess
uv run poe publish
```

## 📂 Structure
- `config.yml`: Configuration for datasets and splits.
- `src/schema_parser.py`: Handles external `tables.json` files.
- `src/schema_generator.py`: Fallback reverse-engineering logic.
- `README.md`: Dataset Card for Hugging Face integration.
