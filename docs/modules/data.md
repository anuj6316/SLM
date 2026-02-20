# Data Pipeline Module

The `slm.data` module handles all data processing: downloading, cleaning, formatting, and splitting datasets.

## Overview

- **Download datasets** from HuggingFace Hub
- **Clean and validate** SQL queries using sqlglot
- **Format to SFT** instruction-input-output format
- **Split datasets** into train/validation sets

## Quick Start

```python
from slm.data import DataPipeline
from slm.config import settings

# Create pipeline
pipeline = DataPipeline(settings)

# Run full pipeline
train_file, val_file = pipeline.run_all()

# Or run individual steps
pipeline.gather()    # Download datasets
pipeline.process()   # Clean + format
pipeline.split()     # Train/val split
```

## Components

### DataPipeline

Main class for orchestrating the data processing workflow.

```python
from slm.data import DataPipeline
from slm.config import settings

pipeline = DataPipeline(settings)

# Download datasets from HuggingFace
paths = pipeline.gather()
# Returns: [Path("data/raw/xlangai_spider_train.jsonl"), ...]

# Process (clean + format)
output = pipeline.process()
# Returns: Path("data/train_sft.jsonl")

# Split into train/val
train_path, val_path = pipeline.split()
# Returns: (Path("data/train_split.jsonl"), Path("data/val_split.jsonl"))

# Run everything
train_path, val_path = pipeline.run_all()
```

### SQLCleaner

Validates and normalizes SQL queries.

```python
from slm.data import SQLCleaner

cleaner = SQLCleaner(dialect="sqlite")

# Normalize SQL
sql = cleaner.normalize_sql("SELECT  *  FROM   users")
# Returns: "SELECT * FROM users"

# Sanitize text
text = cleaner.sanitize_text("  How many   users?  ")
# Returns: "How many users?"

# Process complete record
result = cleaner.process_record(
    question="Find all users",
    sql_query="SELECT * FROM users"
)
# Returns: {"question": "Find all users", "sql": "SELECT * FROM users"}
```

### Download Functions

```python
from slm.data import gather_datasets, download_dataset, fetch_spider_tables
from pathlib import Path

# Download single dataset
path = download_dataset(
    source="xlangai/spider",
    split="train",
    output_dir=Path("data/raw")
)

# Download multiple datasets
paths = gather_datasets(
    datasets=[
        {"source": "xlangai/spider", "split": "train"},
        {"source": "xu3kev/BIRD-SQL-data-train", "split": "train"},
    ],
    output_dir=Path("data/raw"),
    tables_path=Path("data/raw/spider_tables.json")
)

# Fetch Spider tables.json
fetch_spider_tables(Path("data/raw/spider_tables.json"))
```

### Schema Loading

```python
from slm.data import load_schemas
from pathlib import Path

schemas = load_schemas(Path("data/raw/spider_tables.json"))
# Returns:
# {
#   "concert_singer": "singer(id, name, country) | stadium(id, name)",
#   "employee": "employee(id, name, dept_id) | department(id, name)"
# }
```

### Formatting

```python
from slm.data import format_entry, get_dataset_type

# Determine dataset type
dtype = get_dataset_type("xlangai_spider_train.jsonl")
# Returns: "spider"

# Format entry to SFT format
formatted = format_entry(
    entry={
        "question": "How many singers?",
        "cleaned_sql": "SELECT count(*) FROM singer",
        "db_id": "concert_singer"
    },
    dataset_type="spider",
    schema_map={"concert_singer": "singer(id, name)"}
)
# Returns:
# {
#   "instruction": "Convert the following...",
#   "input": "### Database Schema:\nsinger(id, name)\n\n### Question:\nHow many singers?",
#   "output": "SELECT count(*) FROM singer",
#   "metadata": {"dataset": "spider", "db_id": "concert_singer"}
# }
```

### Splitting

```python
from slm.data import split_dataset
from pathlib import Path

train_path, val_path = split_dataset(
    input_file=Path("data/train_sft.jsonl"),
    train_file=Path("data/train_split.jsonl"),
    val_file=Path("data/val_split.jsonl"),
    train_split=0.95,
    seed=42
)
```

## Output Format

Each line in the output JSONL files:

```json
{
  "instruction": "Convert the following natural language question into a valid SQL query based on the provided database schema.",
  "input": "### Database Schema:\nsinger(singer_id, name, country, age)\n\n### Question:\nHow many singers are from USA?",
  "output": "SELECT count(*) FROM singer WHERE country = 'USA'",
  "metadata": {
    "dataset": "spider",
    "db_id": "concert_singer"
  }
}
```

## Supported Datasets

| Dataset | Key Fields |
|---------|------------|
| **Spider** | `query`, `question`, `db_id` |
| **BIRD** | `SQL`, `question`, `db_id` |
| **Gretel** | `sql`, `sql_prompt`, `context` |

## Dataset Type Detection

The pipeline automatically detects dataset type from filename:

- Contains "spider" → `spider`
- Contains "bird" → `bird`
- Contains "gretel" → `gretel`

## SQL Cleaning Process

1. **Parse** - Validate SQL syntax with sqlglot
2. **Normalize** - Convert to canonical single-line form
3. **Dialect** - Ensure SQLite compatibility
4. **Drop invalid** - Remove malformed SQL entries

## Production Tips

1. **Run once, cache results** - Downloaded datasets are cached in `data/raw/`
2. **Validate schemas** - Ensure `tables.json` matches your database
3. **Monitor dropped records** - Check logs for cleaning metrics
4. **Seed reproducibility** - Use fixed seed for consistent splits
5. **Incremental processing** - Files are processed in buffered batches
