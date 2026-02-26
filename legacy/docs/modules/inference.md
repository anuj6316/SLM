# Inference Module

The `slm.inference` module provides SQL generation from natural language questions.

## Overview

- **Single query inference** - Generate SQL for one question
- **Batch inference** - Process entire datasets
- **MLflow tracing** - Track inference with Databricks MLflow
- **Interactive mode** - REPL for testing

## Quick Start

```python
from slm.inference import InferenceEngine
from slm.config import settings

# Create engine
engine = InferenceEngine(settings)

# Generate SQL
result = engine.generate(
    question="How many singers are from USA?",
    db_id="concert_singer"
)

print(result["sql"])  # SELECT count(*) FROM singer WHERE country = 'USA'
```

## InferenceEngine Class

### Initialization

```python
from slm.inference import InferenceEngine
from slm.config import settings
from pathlib import Path

# Using trained model from config
engine = InferenceEngine(settings)

# Using specific model path
engine = InferenceEngine(settings, model_path=Path("outputs/my-model"))

# Using base model (no fine-tuning)
engine = InferenceEngine(settings, model_path=Path("Qwen/Qwen2.5-Coder-3B-Instruct"))
```

### Methods

#### `generate()`

Generate SQL for a single question.

```python
result = engine.generate(
    question="Find all users over 25 years old",
    db_id="employee"
)

# Result structure
{
    "question": "Find all users over 25 years old",
    "db_id": "employee",
    "sql": "SELECT * FROM users WHERE age > 25",
    "raw_output": "SELECT * FROM users WHERE age > 25<|im_end|>",
    "input_tokens": 156,
    "output_tokens": 12,
    "tokens_used": 168
}
```

#### `generate_traced()`

Generate SQL with MLflow tracing.

```python
engine.setup_mlflow()

result = engine.generate_traced(
    question="How many orders?",
    db_id="sales"
)
# Trace logged to Databricks MLflow
```

#### `run_batch()`

Run inference on an entire dataset.

```python
results = engine.run_batch(
    data_path=Path("data/val_split.jsonl"),
    num_samples=100,              # Optional: limit samples
    output_path=Path("results.json"),  # Optional: save results
    use_tracing=True              # Optional: enable MLflow
)

# Returns list of results with exact match
for r in results:
    print(f"Q: {r['question']}")
    print(f"SQL: {r['sql']}")
    print(f"Match: {r['exact_match']}")
```

#### `setup()`

Load model and tokenizer explicitly.

```python
engine.setup()  # Loads model into memory
engine.load_schemas()  # Load database schemas
```

#### `setup_mlflow()`

Configure MLflow for tracing.

```python
engine.setup_mlflow()
# Now all generate_traced() calls are logged
```

## MLflow Tracing

### Setup

1. Configure `.env`:

```bash
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi1234567890abcdef...
```

2. Enable in config:

```yaml
mlflow:
  enabled: true
  tracking_uri: "databricks"
  experiment_name: "/Shared/text2sql"
```

3. Run traced inference:

```python
engine.setup_mlflow()
result = engine.generate_traced(question, db_id)
```

### What Gets Traced

```
generate_sql (LLM) ───────────── 1,245ms
├── Input: 156 tokens
├── Output: 23 tokens
└── SQL: SELECT count(*) FROM singer
```

## Inference Configuration

```yaml
inference:
  data_path: "data/val_split.jsonl"
  max_new_tokens: 128
  output_path: null
  num_samples: null
  do_sample: false
  temperature: 0.0
```

## CLI Usage

### Batch Inference

```bash
# Run on validation set
python main.py infer

# Limit samples
python main.py infer --num-samples 100

# Custom dataset
python main.py infer --data-path data/test.jsonl

# Save results
python main.py infer --output-path results/inference.json

# With MLflow tracing
python main.py infer --trace
```

### Interactive Mode

```bash
python main.py infer -i

# Output:
# Text-to-SQL Inference (type 'quit' to exit)
# ----------------------------------------
#
# Question: How many users?
# DB ID (default: concert_singer): employee
#
# SQL: SELECT count(*) FROM users
# Tokens: 145
```

## Output Format

Batch inference saves JSON:

```json
{
  "metadata": {
    "model": "outputs/qwen-coder-3b-text2sql",
    "dataset": "data/val_split.jsonl",
    "num_samples": 100,
    "accuracy": 0.72,
    "timestamp": "2026-02-20T16:00:00"
  },
  "results": [
    {
      "question": "How many singers?",
      "db_id": "concert_singer",
      "sql": "SELECT count(*) FROM singer",
      "gold_sql": "SELECT count(*) FROM singer",
      "exact_match": true,
      "tokens_used": 145
    }
  ]
}
```

## Production Tips

1. **Pre-load model** - Call `setup()` before first request
2. **Batch requests** - Use `run_batch()` for efficiency
3. **Enable tracing** - Track performance in production
4. **Set token limits** - Prevent runaway generation
5. **Cache schemas** - `load_schemas()` is called once
6. **Use greedy decoding** - `do_sample: false` for consistency

## Performance Optimization

### For High Throughput

```python
# Pre-load everything
engine.setup()
engine.load_schemas()

# Process in batches
results = engine.run_batch(data_path, use_tracing=False)
```

### For Low Latency

```python
# Use smaller model
settings.model.name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Reduce max tokens
settings.inference.max_new_tokens = 64
```

## Troubleshooting

### Slow First Inference

- First inference loads model (takes 10-30s)
- Call `setup()` at application startup

### Incorrect SQL

- Check schema is correct in `tables.json`
- Verify question is clear and unambiguous
- Try increasing `max_new_tokens`

### MLflow Connection Errors

- Verify `DATABRICKS_HOST` and `DATABRICKS_TOKEN`
- Check network connectivity
- Ensure experiment exists in Databricks
