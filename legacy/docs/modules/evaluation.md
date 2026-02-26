# Evaluation Module

The `slm.evaluation` module evaluates Text-to-SQL models on accuracy metrics.

## Overview

- **Exact match accuracy** - Compare predicted SQL to gold SQL
- **Spider benchmark** - Official Spider evaluation
- **Batch evaluation** - Process entire datasets
- **Model comparison** - Compare multiple models

## Quick Start

```python
from slm.evaluation import Evaluator
from slm.config import settings

# Create evaluator
evaluator = Evaluator(settings)

# Evaluate on validation set
results = evaluator.evaluate_dataset("data/val_split.jsonl")

print(f"Accuracy: {results['metrics']['exact_match_accuracy']:.2%}")
```

## Evaluator Class

### Initialization

```python
from slm.evaluation import Evaluator
from slm.config import settings
from pathlib import Path

# Using trained model from config
evaluator = Evaluator(settings)

# Using specific model path
evaluator = Evaluator(settings, model_path=Path("outputs/my-model"))
```

### Methods

#### `evaluate_dataset()`

Evaluate model on a JSONL dataset.

```python
results = evaluator.evaluate_dataset(
    data_path=Path("data/val_split.jsonl"),
    output_path=Path("eval_results/eval.json")  # Optional
)

# Results structure
{
    "metadata": {
        "model": "outputs/qwen-coder-3b-text2sql",
        "dataset": "data/val_split.jsonl",
        "num_samples": 5818,
        "timestamp": "2026-02-20T16:00:00"
    },
    "metrics": {
        "exact_match_accuracy": 0.72,
        "exact_matches": 4189,
        "total_samples": 5818
    },
    "per_sample": [
        {
            "db_id": "concert_singer",
            "prediction": "SELECT count(*) FROM singer",
            "gold": "SELECT count(*) FROM singer",
            "exact_match": True
        },
        ...
    ]
}
```

#### `evaluate_spider()`

Evaluate on Spider benchmark with official evaluation.

```python
results = evaluator.evaluate_spider(
    dev_path=Path("data/raw/spider_dev.json"),
    db_path=Path("data/spider/database"),
    spider_eval_script=Path("scripts/spider/evaluation.py")
)
```

#### `compare_models()`

Compare multiple models on the same dataset.

```python
results = evaluator.compare_models(
    model_paths=[
        Path("outputs/model-v1"),
        Path("outputs/model-v2"),
        Path("outputs/model-v3"),
    ],
    data_path=Path("data/val_split.jsonl")
)

# Saves comparison to eval_results/model_comparison.json
```

## Metrics Functions

```python
from slm.evaluation import calculate_exact_match, extract_question

# Calculate accuracy
matches, accuracy = calculate_exact_match(
    predictions=["SELECT * FROM users", "SELECT id FROM users"],
    gold_queries=["SELECT * FROM users", "SELECT id FROM user"]
)
# Returns: (1, 0.5)

# Extract question from formatted input
question = extract_question({
    "input": "### Database Schema:\nusers(id)\n\n### Question:\nHow many users?"
})
# Returns: "How many users?"
```

## Evaluation Output

Results are saved to `eval_results/`:

```
eval_results/
├── model_comparison.json      # Multi-model comparison
├── qwen-coder-3b_eval.json    # Single model evaluation
├── predictions.txt            # Raw predictions
└── gold.txt                   # Gold queries
```

## CLI Usage

```bash
# Evaluate with default settings
python main.py eval

# Evaluate specific model
python main.py eval --model-path outputs/my-model

# Evaluate on custom dataset
python main.py eval --data-path data/test.jsonl

# Save results to file
python main.py eval --output-path results/eval.json
```

## Accuracy Interpretation

| Accuracy | Quality | Description |
|----------|---------|-------------|
| < 50% | Poor | Model needs more training data |
| 50-70% | Fair | Basic queries working |
| 70-85% | Good | Production-ready for simple queries |
| 85-95% | Excellent | Handles complex queries |
| > 95% | Outstanding | Near-human performance |

## Production Tips

1. **Evaluate regularly** - Track accuracy over training
2. **Use held-out data** - Never evaluate on training data
3. **Analyze failures** - Review `per_sample` for error patterns
4. **Compare baselines** - Always compare to previous models
5. **Track by dataset** - Spider vs BIRD vs Gretel accuracy
