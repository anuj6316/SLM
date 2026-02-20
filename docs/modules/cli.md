# CLI Module

The `slm.cli` module provides the command-line interface for the entire pipeline.

## Overview

- **Unified entry point** - Single `main.py` for all operations
- **Subcommands** - data, train, eval, infer, pipeline
- **Argument parsing** - Full control via CLI flags

## Quick Start

```bash
# Show help
python main.py --help

# Data pipeline
python main.py data all

# Training
python main.py train sft

# Evaluation
python main.py eval

# Inference
python main.py infer
```

## Commands

### data

Data pipeline operations.

```bash
# Download datasets
python main.py data gather

# Process (clean + format)
python main.py data process

# Split train/val
python main.py data split

# Run all steps
python main.py data all
```

### train

Training operations.

```bash
# SFT training
python main.py train sft

# GRPO training (not implemented)
python main.py train grpo
```

### eval

Evaluation operations.

```bash
# Evaluate on validation set
python main.py eval

# Custom dataset
python main.py eval --data-path data/test.jsonl

# Specific model
python main.py eval --model-path outputs/my-model

# Save results
python main.py eval --output-path results/eval.json
```

### infer

Inference operations.

```bash
# Batch inference
python main.py infer

# Interactive mode
python main.py infer -i

# With MLflow tracing
python main.py infer --trace

# Limit samples
python main.py infer --num-samples 100

# Custom dataset
python main.py infer --data-path data/test.jsonl

# Custom model
python main.py infer --model-path outputs/my-model

# Save results
python main.py infer --output-path results/inference.json
```

### pipeline

Full pipeline operations.

```bash
# Run complete pipeline
python main.py pipeline full

# Train + evaluate
python main.py pipeline sft-eval
```

## Global Options

```bash
# Use custom config
python main.py --config custom_config.yaml data all
```

## Poe Tasks

Convenient shortcuts defined in `pyproject.toml`:

```bash
# Using poe
uv run poe data          # python main.py data all
uv run poe train         # python main.py train sft
uv run poe eval          # python main.py eval
uv run poe infer         # python main.py infer
uv run poe infer-trace   # python main.py infer --trace
uv run poe pipeline      # python main.py pipeline full
uv run poe mlflow-ui     # mlflow ui --port 5000
```

## Programmatic Usage

```python
from slm.cli import main
import sys

# Run CLI programmatically
sys.argv = ["main.py", "data", "all"]
main()
```

## Production Tips

1. **Use poe tasks** - Shorter commands for common operations
2. **Version configs** - Track which config produced which model
3. **Log outputs** - Redirect to files for debugging
4. **Use --trace** - Enable MLflow for production inference
