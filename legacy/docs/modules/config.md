# Configuration Module

The `slm.config` module provides centralized configuration management using Pydantic Settings.

## Overview

- **Type-safe configuration** with Pydantic validation
- **YAML file loading** from `config.yaml`
- **Environment variable overrides** with `SLM__` prefix
- **Nested settings** for organized configuration

## Quick Start

```python
from slm.config import settings

# Access configuration
print(settings.model.name)           # Qwen/Qwen2.5-Coder-3B-Instruct
print(settings.training.epochs)     # 3
print(settings.data.train_split)    # 0.95
```

## Configuration Sections

### Project Settings

```yaml
project:
  name: "SLM Text-to-SQL"
  version: "1.0.0"
```

```python
settings.project.name     # "SLM Text-to-SQL"
settings.project.version  # "1.0.0"
```

### Model Settings

```yaml
model:
  name: "Qwen/Qwen2.5-Coder-3B-Instruct"
  max_seq_length: 2048
  load_in_4bit: true
  dtype: null  # Auto-detect
```

```python
settings.model.name           # Model name or path
settings.model.max_seq_length # Maximum sequence length
settings.model.load_in_4bit   # Use 4-bit quantization
```

### LoRA Settings

```yaml
lora:
  r: 64
  lora_alpha: 64
  lora_dropout: 0.0
  bias: "none"
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

### Training Settings

```yaml
training:
  output_dir: "outputs/qwen-coder-3b-text2sql"
  epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 0.0002
  lr_scheduler_type: "cosine"
  warmup_steps: 50
  weight_decay: 0.01
  logging_steps: 10
  save_steps: 500
  eval_strategy: "steps"
  eval_steps: 500
  save_total_limit: 2
  optim: "adamw_8bit"
  report_to: "mlflow"
```

### Data Settings

```yaml
data:
  datasets:
    - name: "spider"
      source: "xlangai/spider"
      split: "train"
    - name: "bird"
      source: "xu3kev/BIRD-SQL-data-train"
      split: "train"
  output_dir: "data"
  raw_dir: "data/raw"
  train_file: "data/train_split.jsonl"
  val_file: "data/val_split.jsonl"
  tables_file: "data/raw/spider_tables.json"
  train_split: 0.95
  seed: 42
```

### MLflow Settings

```yaml
mlflow:
  enabled: true
  tracking_uri: "databricks"
  experiment_name: "/Shared/text2sql"
  registry_uri: "databricks-uc"
```

### Inference Settings

```yaml
inference:
  data_path: "data/val_split.jsonl"
  max_new_tokens: 128
  output_path: null
  num_samples: null
  do_sample: false
  temperature: 0.0
```

### Formatting Settings

```yaml
formatting:
  system_prompt: "You are an expert Text-to-SQL assistant..."
  chat_template: "chatml"
```

## Environment Variable Overrides

Override any setting using environment variables with `SLM__` prefix:

```bash
# Override model name
export SLM__MODEL__NAME="Qwen/Qwen2.5-Coder-7B-Instruct"

# Override training epochs
export SLM__TRAINING__EPOCHS=5

# Override learning rate
export SLM__TRAINING__LEARNING_RATE=0.0001
```

## Loading Custom Configuration

```python
from slm.config import Settings

# Load from custom path
settings = Settings.from_yaml("path/to/config.yaml")

# Or use default path
settings = Settings.from_yaml()  # Loads from config.yaml
```

## Available Classes

| Class | Description |
|-------|-------------|
| `Settings` | Main configuration class |
| `ProjectSettings` | Project metadata |
| `ModelSettings` | Model configuration |
| `LoRASettings` | LoRA/PEFT configuration |
| `TrainingSettings` | Training hyperparameters |
| `DataSettings` | Data pipeline configuration |
| `MLflowSettings` | MLflow tracking configuration |
| `EvaluationSettings` | Evaluation configuration |
| `InferenceSettings` | Inference configuration |
| `FormattingSettings` | Prompt formatting configuration |
| `DatasetConfig` | Individual dataset configuration |

## Production Tips

1. **Use environment variables for secrets** - Never commit API keys to config files
2. **Version your config files** - Track configuration changes in git
3. **Validate on startup** - Pydantic will catch configuration errors early
4. **Use different configs for environments** - `config.dev.yaml`, `config.prod.yaml`
