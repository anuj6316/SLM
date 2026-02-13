# 🚀 SLM Preprocessing & SFT Pipeline

A high-performance, streaming-first pipeline designed for fine-tuning Small Language Models (SLMs) like Qwen, Llama, and Mistral. This toolkit handles everything from raw data ingestion to generating a fine-tuned model ready for deployment.

---

## 🏗️ Architecture Overview

The project is divided into two main phases:
1.  **Preprocessing Phase**: Standardizes raw data (CSV, Hugging Face) into a clean SFT format.
2.  **SFT Module**: Fine-tunes the model using **Unsloth** (2x faster, 70% less VRAM).

---

## 🛠️ Phase 1: Preprocessing

The preprocessing module uses a **Streaming-first** approach, meaning it can handle datasets larger than your RAM by processing one row at a time.

### 1. Supported Loaders
- **CSV Loader**: Fast, native processing for local `.csv` files.
- **HF Loader**: Seamless integration with the Hugging Face `datasets` library (Local or Remote).

### 2. The SFT Processor
The `SFTProcessor` maps your raw columns to the standard format required for training:
- `instruction`: The task or question.
- `input`: The context (supports combining multiple columns into one string).
- `output`: The desired response.

### 3. Usage
Configure your dataset in `config.yaml` and run:
```bash
uv run src/preprocess.py
```

---

## 🏋️ Phase 2: Supervised Fine-Tuning (SFT)

The training module leverages **Unsloth** and **PEFT (LoRA)** to make training efficient and accessible on consumer hardware.

### Key Features
- **4-bit Quantization**: Train large models on smaller GPUs.
- **ChatML Template**: Automatically formats your instructions into a conversational style.
- **LoRA Adapters**: Only trains a fraction of the model parameters, saving time and memory.

### Usage
Ensure your preprocessed data is ready, then run:
```bash
uv run src/train.py
```

---

## ⚙️ Configuration (`config.yaml`)

The entire pipeline is controlled by a single configuration file. Here is how to set up your data mapping:

```yaml
data:
  source_type: "csv"
  path: "data/my_data.csv"
  columns:
    instruction: "User_Question"     # Map single column
    input: ["Context", "Schema"]     # Combine multiple columns
    output: "SQL_Query"              # Map target column

training:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  epochs: 3
  batch_size: 2
  learning_rate: 2e-4
```

---

## 🧪 Quality Assurance (QA)

This codebase has been audited for edge cases. It includes:
- **Lenient Mode**: Skips malformed rows without stopping the pipeline.
- **Strict Mode**: Optional toggle to fail fast if data quality is a priority.
- **Detailed Logging**: Change `level: DEBUG` in `preprocess.py` for a row-by-row execution trace.

---

## 📂 Project Structure

- `src/preprocessing/loaders`: Logic for fetching data.
- `src/preprocessing/processors`: Logic for mapping and cleaning.
- `src/preprocessing/formatters`: Logic for saving (JSONL).
- `src/train.py`: The Unsloth training engine.
- `config.yaml`: The central control hub.

---

## 🚀 Quick Start (Titanic Example)
1. Update `config.yaml` to point to `data/Titanic-Dataset.csv`.
2. Map `Name` -> `instruction`, `["Sex", "Age"]` -> `input`, and `Survived` -> `output`.
3. Run `uv run src/preprocess.py`.
4. Check `data/train_sft.jsonl` for the results.
5. Run `uv run src/train.py` to start training!
