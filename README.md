# SLM: Text-to-SQL Pipeline

A production-grade pipeline for fine-tuning Small Language Models (SLMs) on Text-to-SQL tasks. This project implements a full MLOps lifecycle, including automated data gathering, SQL cleaning, reverse-engineered schema anchoring, and Hugging Face integration.

## 🏗 Architecture
![Text-to-SQL MLOps Pipeline Flow](https://storage.googleapis.com/second-petal-295822.appspot.com/elements/elements%3Ad6a8a796d7c03b962ceaf006140dbc0218daf65f7ac9c3b5595641b2c787654b.png)

- **Orchestrator:** Main entry point for pipeline stages.
- **Preprocessing:** Modular system to gather, clean, and format datasets (Spider, BIRD, Gretel).
- **Versioning:** Automated data versioning using Hugging Face Hub tags.
- **Training:** Efficient fine-tuning using Unsloth and SFTTrainer.

## 🚀 Quick Start

### 1. Setup Environment
Ensure you have `uv` installed, then:
```bash
uv sync
```
Create a `.env` file with your `HF_TOKEN`.

### 2. Preprocess Data
Gathers raw data, standardizes SQL via `sqlglot`, and reverse-engineers database schemas.
```bash
uv run poe preprocess
```

### 3. Publish to Hugging Face
Uploads the "Gold Standard" SFT dataset to your HF repository with an automated version tag.
```bash
uv run poe publish
```

### 4. Train Model
Starts the fine-tuning process using the configuration in `config.yaml`.
```bash
uv run python main.py train
```

## 🛠 MLOps Standards
- **Registry:** Hugging Face Hub (Data & Models).
- **Tracking:** Weights & Biases / MLflow integration.
- **Automation:** Poe the Poet task runner.
- **Security:** Git history purged of large artifacts and secrets.
