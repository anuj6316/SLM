# Training Module

The `slm.training` module provides Supervised Fine-Tuning (SFT) for Text-to-SQL models using Unsloth and LoRA.

## Overview

- **Efficient training** with Unsloth optimizations (2x faster, 70% less VRAM)
- **LoRA/PEFT** for parameter-efficient fine-tuning
- **ChatML formatting** for instruction-following
- **MLflow integration** for experiment tracking

## Quick Start

```python
from slm.training import SFTTrainer
from slm.config import settings

# Create trainer
trainer = SFTTrainer(settings)

# Train model
output_dir = trainer.train()
print(f"Model saved to: {output_dir}")
```

## SFTTrainer Class

### Initialization

```python
from slm.training import SFTTrainer
from slm.config import Settings

# Using global settings
trainer = SFTTrainer(settings)

# Using custom settings
custom_settings = Settings.from_yaml("custom_config.yaml")
trainer = SFTTrainer(custom_settings)
```

### Methods

#### `train()`

Run the full SFT training pipeline.

```python
output_dir = trainer.train()
# Returns: Path to output directory
```

**What happens:**
1. Loads base model with Unsloth
2. Injects LoRA adapters
3. Loads and formats training data
4. Runs SFT training
5. Saves LoRA adapters

#### `setup()`

Load model and tokenizer separately (for advanced use).

```python
trainer.setup()
# Now trainer._model and trainer._tokenizer are loaded
```

#### `save_merged()`

Save merged model (base + LoRA) for deployment.

```python
merged_path = trainer.save_merged()
# Or specify custom path
merged_path = trainer.save_merged(Path("outputs/merged_model"))
```

## Training Configuration

### Model Settings

```yaml
model:
  name: "Qwen/Qwen2.5-Coder-3B-Instruct"
  max_seq_length: 2048
  load_in_4bit: true    # Enable for limited VRAM
  dtype: null           # Auto-detect
```

### LoRA Settings

```yaml
lora:
  r: 64              # LoRA rank
  lora_alpha: 64     # Scaling factor
  lora_dropout: 0.0  # Dropout rate
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

**Choosing LoRA rank:**

| GPU VRAM | Recommended `r` |
|----------|------------------|
| 8 GB | 16-32 |
| 12 GB | 32-48 |
| 16 GB+ | 64 |

### Training Hyperparameters

```yaml
training:
  epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8   # Effective batch = 2 × 8 = 16
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

## Training Data Format

The trainer expects JSONL files with this structure:

```json
{
  "instruction": "Convert the question to SQL...",
  "input": "### Database Schema:\nusers(id, name)\n\n### Question:\nHow many users?",
  "output": "SELECT count(*) FROM users"
}
```

## ChatML Formatting

Training examples are formatted to ChatML:

```
<|im_start|>system
You are an expert Text-to-SQL assistant...<|im_end|>
<|im_start|>user
### Database Schema:
users(id, name, age)

### Question:
How many users are over 25?<|im_end|>
<|im_start|>assistant
SELECT count(*) FROM users WHERE age > 25<|im_end|>
```

## Hardware Requirements

| Model Size | VRAM (4-bit) | VRAM (16-bit) |
|------------|--------------|---------------|
| 1.5B | 6 GB | 8 GB |
| 3B | 8 GB | 12 GB |
| 7B | 12 GB | 20 GB |

## Training Output

After training, the output directory contains:

```
outputs/qwen-coder-3b-text2sql/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── trainer_state.json
```

## CLI Usage

```bash
# Train with default config
python main.py train sft

# Train with custom config
python main.py train sft --config custom_config.yaml
```

## Environment Variables

```bash
# Override settings via environment
export SLM__TRAINING__EPOCHS=5
export SLM__TRAINING__LEARNING_RATE=0.0001
export SLM__MODEL__LOAD_IN_4BIT=true
```

## Monitoring Training

### With MLflow

```bash
# Start MLflow UI
mlflow ui --port 5000

# View at http://localhost:5000
```

### Training Metrics

- `train_loss` - Training loss per step
- `eval_loss` - Validation loss per eval
- `learning_rate` - Current learning rate

## Production Tips

1. **Start with small epochs** - 2-3 epochs usually sufficient
2. **Monitor eval_loss** - Early stop if not improving
3. **Save checkpoints** - `save_total_limit: 2` keeps best checkpoints
4. **Use 4-bit for experimentation** - Faster iteration
5. **Validate on held-out set** - Don't use training data for eval
6. **Log experiments** - Enable MLflow for reproducibility

## Troubleshooting

### CUDA Out of Memory

```yaml
# Reduce batch size
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16

# Reduce LoRA rank
lora:
  r: 32

# Enable 4-bit
model:
  load_in_4bit: true
```

### Slow Training

- Enable `load_in_4bit: true`
- Use `optim: "adamw_8bit"`
- Increase `gradient_accumulation_steps`

### Model Not Learning

- Check learning rate (try 1e-4 to 5e-4)
- Verify data format is correct
- Ensure schema is present in training data
