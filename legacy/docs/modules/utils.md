# Utils Module

The `slm.utils` module provides shared utilities used across the pipeline.

## Overview

- **Logging setup** - Rich-formatted logging
- **Schema loading** - Load database schemas from tables.json
- **SQL cleaning** - Clean and normalize SQL output

## Quick Start

```python
from slm.utils import setup_logging, load_schema_dict, clean_sql

# Setup logging
logger = setup_logging("my_module")

# Load schemas
schemas = load_schema_dict(Path("data/raw/spider_tables.json"))

# Clean SQL output
sql = clean_sql("SELECT * FROM users<|im_end|>")
```

## Logging

### setup_logging()

Configure logging with Rich handler.

```python
from slm.utils import setup_logging

logger = setup_logging("my_module")
logger.info("Processing started")
logger.warning("Low memory")
logger.error("Failed to process")
```

### get_logger()

Get a logger for a module.

```python
from slm.utils import get_logger

logger = get_logger(__name__)
logger.info("Module loaded")
```

## Schema Loading

### load_schema_dict()

Load schemas from Spider/BIRD style tables.json.

```python
from slm.utils import load_schema_dict
from pathlib import Path

schemas = load_schema_dict(Path("data/raw/spider_tables.json"))

# Result:
# {
#   "concert_singer": "singer(singer_id, name, country, age) | stadium(stadium_id, name, capacity)",
#   "employee": "employee(id, name, dept_id, salary) | department(dept_id, name, location)"
# }
```

**Input format (tables.json):**

```json
[
  {
    "db_id": "concert_singer",
    "table_names_original": ["singer", "stadium"],
    "column_names_original": [
      [-1, "*"],
      [0, "singer_id"],
      [0, "name"],
      [1, "stadium_id"]
    ]
  }
]
```

## SQL Utilities

### clean_sql()

Clean generated SQL output.

```python
from slm.utils import clean_sql

# Remove markdown code blocks
sql = clean_sql("```sql\nSELECT * FROM users\n```")
# Returns: "SELECT * FROM users"

# Remove end markers
sql = clean_sql("SELECT * FROM users<|im_end|>")
# Returns: "SELECT * FROM users"

# Normalize whitespace
sql = clean_sql("SELECT   *   FROM   users")
# Returns: "SELECT * FROM users"
```

### extract_question_from_input()

Extract question from formatted input.

```python
from slm.utils import extract_question_from_input

input_text = """### Database Schema:
users(id, name, age)

### Question:
How many users are over 25?
"""

question = extract_question_from_input(input_text)
# Returns: "How many users are over 25?"
```

## Usage in Modules

### In Data Pipeline

```python
from slm.utils import load_schema_dict

schemas = load_schema_dict(settings.data.tables_file)
```

### In Inference

```python
from slm.utils import clean_sql, load_schema_dict

# Clean model output
sql = clean_sql(raw_output)

# Load schemas
schemas = load_schema_dict(tables_path)
```

### In Evaluation

```python
from slm.utils import extract_question_from_input

question = extract_question_from_input(entry["input"])
```

## Production Tips

1. **Use module loggers** - `get_logger(__name__)` for each module
2. **Cache schemas** - Load once, reuse across inference calls
3. **Validate cleaned SQL** - Check output isn't empty after cleaning
