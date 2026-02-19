---
license: apache-2.0
task_categories:
- text-to-sql
language:
- en
tags:
- text2sql
- sft
- fine-tuning
size_categories:
- 100K<n<1M
---

# Text2SQL SFT Dataset

This dataset is a combined and cleaned version of Spider, BIRD, and Gretel datasets, formatted specifically for Instruction Fine-Tuning of Small Language Models (SLMs).

## Dataset Structure
Each record contains:
- **instruction**: The task description.
- **input**: The Database Schema and the Natural Language Question.
- **output**: The canonicalized SQL query.
- **metadata**: A dictionary containing `dataset` source and `db_id`.

## Usage
```python
from datasets import load_dataset
dataset = load_dataset("anuj6316/text2sql")
```
