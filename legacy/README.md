# SLM Text-to-SQL

> Production-ready Text-to-SQL pipeline for fine-tuning Small Language Models with MLflow observability.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Data Pipeline](#data-pipeline)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Inference](#inference)
9. [MLflow Tracing](#mlflow-tracing)
10. [CLI Reference](#cli-reference)
11. [Python API](#python-api)
12. [Production Deployment](#production-deployment)
13. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Run data pipeline
uv run poe data

# 3. Train the model
uv run poe train

# 4. Evaluate
uv run poe eval

# 5. Run inference
uv run poe infer

# Or run everything
uv run poe pipeline
```

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Required for type hints |
| CUDA GPU | 8GB+ VRAM | 12GB+ recommended |
| uv | Latest | [Install here](https://github.com/astral-sh/uv) |

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/slm-text2sql.git
cd slm-text2sql

# Install dependencies
uv sync

# Create .env file
cp .env.example .env
# Edit .env with your credentials

# Verify installation
uv run python -c "from slm import settings; print(settings.model.name)"
```

### Environment Variables

```bash
# .env
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi1234567890abcdef...
```

---

## Project Structure

```
slm/
├── config/                 # Configuration management
│   ├── __init__.py
│   └── settings.py         # Pydantic settings classes
│
├── data/                   # Data pipeline
│   ├── __init__.py
│   ├── pipeline.py         # DataPipeline orchestrator
│   ├── cleaner.py          # SQL validation
│   ├── gatherer.py         # Dataset download
│   ├── formatter.py        # SFT formatting
│   ├── splitter.py         # Train/val split
│   └── schemas.py          # Schema loading
│
├── training/               # Training pipeline
│   ├── __init__.py
│   └── trainer.py          # SFTTrainer class
│
├── evaluation/             # Evaluation pipeline
│   ├── __init__.py
│   ├── evaluator.py        # Evaluator class
│   └── metrics.py          # Accuracy metrics
│
├── inference/              # Inference pipeline
│   ├── __init__.py
│   ├── engine.py           # InferenceEngine class
│   └── tracing.py          # MLflow tracing
│
├── utils/                  # Shared utilities
│   ├── __init__.py
│   ├── logging.py
│   ├── schema.py
│   └── sql.py
│
├── cli/                    # Command-line interface
│   ├── __init__.py
│   └── commands.py
│
├── __init__.py             # Public API exports
└── py.typed                # PEP 561 marker
```

---

## Configuration

### Configuration File

All settings in `config.yaml`:

```yaml
# Model
model:
  name: "Qwen/Qwen2.5-Coder-3B-Instruct"
  max_seq_length: 2048
  load_in_4bit: true

# LoRA
lora:
  r: 64
  lora_alpha: 64
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

# Training
training:
  epochs: 3
  learning_rate: 0.0002
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8

# MLflow
mlflow:
  enabled: true
  tracking_uri: "databricks"
  experiment_name: "/Shared/text2sql"

# Data
data:
  train_split: 0.95
  seed: 42
```

### Environment Overrides

```bash
export SLM__MODEL__NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
export SLM__TRAINING__EPOCHS=5
export SLM__TRAINING__LEARNING_RATE=0.0001
```

### Python API

```python
from slm.config import settings, Settings

# Access settings
print(settings.model.name)
print(settings.training.epochs)

# Custom config
settings = Settings.from_yaml("custom_config.yaml")
```

**See:** [docs/modules/config.md](docs/modules/config.md)

---

## Data Pipeline

### CLI

```bash
# Full pipeline
python main.py data all

# Individual steps
python main.py data gather   # Download from HuggingFace
python main.py data process  # Clean + format
python main.py data split    # Train/val split
```

### Python API

```python
from slm.data import DataPipeline
from slm.config import settings

pipeline = DataPipeline(settings)

# Run all steps
train_path, val_path = pipeline.run_all()

# Individual steps
pipeline.gather()
pipeline.process()
pipeline.split()
```

### Output Format

```json
{
  "instruction": "Convert the following natural language question...",
  "input": "### Database Schema:\nusers(id, name)\n\n### Question:\nHow many users?",
  "output": "SELECT count(*) FROM users",
  "metadata": {"dataset": "spider", "db_id": "employee"}
}
```

### Supported Datasets

| Dataset | HuggingFace | Samples |
|---------|-------------|---------|
| Spider | xlangai/spider | 7,000 |
| BIRD | xu3kev/BIRD-SQL-data-train | 9,428 |
| Gretel | gretelai/synthetic_text_to_sql | 99,927 |

**See:** [docs/modules/data.md](docs/modules/data.md)

---

## Training

### CLI

```bash
# SFT training
python main.py train sft

# With custom config
python main.py train sft --config custom_config.yaml
```

### Python API

```python
from slm.training import SFTTrainer
from slm.config import settings

trainer = SFTTrainer(settings)
output_dir = trainer.train()

# Save merged model
trainer.save_merged()
```

### Hardware Requirements

| Model | VRAM (4-bit) | VRAM (16-bit) |
|-------|--------------|---------------|
| 1.5B | 6 GB | 8 GB |
| 3B | 8 GB | 12 GB |
| 7B | 12 GB | 20 GB |

### Training Output

```
outputs/qwen-coder-3b-text2sql/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
└── trainer_state.json
```

**See:** [docs/modules/training.md](docs/modules/training.md)

---

## Evaluation

### CLI

```bash
# Evaluate on validation set
python main.py eval

# Custom options
python main.py eval --model-path outputs/my-model --data-path data/test.jsonl
```

### Python API

```python
from slm.evaluation import Evaluator
from slm.config import settings

evaluator = Evaluator(settings)
results = evaluator.evaluate_dataset("data/val_split.jsonl")

print(f"Accuracy: {results['metrics']['exact_match_accuracy']:.2%}")
```

### Output

```json
{
  "metrics": {
    "exact_match_accuracy": 0.72,
    "exact_matches": 4189,
    "total_samples": 5818
  },
  "per_sample": [...]
}
```

**See:** [docs/modules/evaluation.md](docs/modules/evaluation.md)

---

## Inference

### CLI

```bash
# Batch inference
python main.py infer

# Interactive mode
python main.py infer -i

# With MLflow tracing
python main.py infer --trace

# Limit samples
python main.py infer --num-samples 100
```

### Python API

```python
from slm.inference import InferenceEngine
from slm.config import settings

engine = InferenceEngine(settings)

# Single query
result = engine.generate(
    question="How many users?",
    db_id="employee"
)
print(result["sql"])

# Batch inference
results = engine.run_batch(
    data_path="data/val_split.jsonl",
    num_samples=100
)
```

### Output

```python
{
    "question": "How many users?",
    "sql": "SELECT count(*) FROM users",
    "tokens_used": 145
}
```

**See:** [docs/modules/inference.md](docs/modules/inference.md)

---

## MLflow Tracing

### Setup

1. Configure `.env`:
```bash
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
```

2. Enable in config:
```yaml
mlflow:
  enabled: true
  tracking_uri: "databricks"
  experiment_name: "/Shared/text2sql"
```

3. Run traced inference:
```bash
python main.py infer --trace
```

### View Traces

1. Open Databricks workspace
2. Go to **AI/ML** → **Experiments**
3. Select experiment `/Shared/text2sql`
4. Click **Traces** tab

### What Gets Traced

```
generate_sql (LLM) ───────────── 1,245ms
├── Input: 156 tokens
├── Output: 23 tokens
└── SQL: SELECT count(*) FROM singer
```

---

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `python main.py data all` | Full data pipeline |
| `python main.py train sft` | SFT training |
| `python main.py eval` | Evaluate model |
| `python main.py infer` | Batch inference |
| `python main.py infer -i` | Interactive mode |
| `python main.py infer --trace` | Traced inference |
| `python main.py pipeline full` | Complete pipeline |

### Poe Tasks

```bash
uv run poe data          # Data pipeline
uv run poe train         # Training
uv run poe eval          # Evaluation
uv run poe infer         # Inference
uv run poe infer-trace   # Traced inference
uv run poe pipeline      # Full pipeline
uv run poe mlflow-ui     # Start MLflow UI
```

---

## Python API

### Quick Reference

```python
from slm import settings, DataPipeline, SFTTrainer, Evaluator, InferenceEngine

# Data
pipeline = DataPipeline(settings)
train_path, val_path = pipeline.run_all()

# Training
trainer = SFTTrainer(settings)
output_dir = trainer.train()

# Evaluation
evaluator = Evaluator(settings)
results = evaluator.evaluate_dataset("data/val_split.jsonl")

# Inference
engine = InferenceEngine(settings)
result = engine.generate("How many users?", "employee")
```

---

## Production Deployment

### Checklist

- [ ] Configure `.env` with production credentials
- [ ] Set `mlflow.enabled: true` for observability
- [ ] Use `load_in_4bit: false` for inference speed
- [ ] Pre-load model at startup with `engine.setup()`
- [ ] Enable MLflow tracing for monitoring
- [ ] Set appropriate `max_new_tokens` limit
- [ ] Cache schemas with `engine.load_schemas()`

### Performance Tuning

```yaml
# For high throughput
inference:
  max_new_tokens: 64
  do_sample: false

# For low latency
model:
  load_in_4bit: false  # Full precision is faster
```

### API Server Example

```python
from fastapi import FastAPI
from slm.inference import InferenceEngine
from slm.config import settings

app = FastAPI()
engine = InferenceEngine(settings)
engine.setup()
engine.load_schemas()

@app.post("/generate")
def generate(question: str, db_id: str):
    result = engine.generate(question, db_id)
    return {"sql": result["sql"]}
```

### Docker

```dockerfile
FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install uv && uv sync

CMD ["python", "main.py", "infer", "--trace"]
```

---

## Troubleshooting

### CUDA Out of Memory

```yaml
# Reduce batch size
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16

# Enable 4-bit
model:
  load_in_4bit: true

# Reduce LoRA rank
lora:
  r: 32
```

### Slow Training

- Enable `load_in_4bit: true`
- Use `optim: "adamw_8bit"`
- Increase `gradient_accumulation_steps`

### MLflow Connection Errors

- Verify `DATABRICKS_HOST` and `DATABRICKS_TOKEN`
- Check network connectivity
- Ensure experiment exists

### Incorrect SQL Generation

- Verify schema in `tables.json`
- Check question clarity
- Increase `max_new_tokens`

### Model Not Learning

- Check learning rate (1e-4 to 5e-4)
- Verify data format
- Ensure schema is in training data

---

## Expected Performance

| Model | Spider (SFT) | Spider (SFT+GRPO) | VRAM |
|-------|--------------|-------------------|------|
| Qwen2.5-Coder-3B | 72-75% | 80-84% | 8 GB |
| Qwen2.5-Coder-7B | 85-88% | 90-93% | 16 GB |

---

## License

MIT License
