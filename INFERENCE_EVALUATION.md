# Inference & Evaluation Guide

This guide explains how to generate SQL predictions from your fine-tuned models and evaluate their accuracy against a test set.

---

## 🚀 Running Inference

To test a model, use the `test-model` task. This script loads your fine-tuned model (or a base model) and runs it against a test dataset.

### Basic Usage

```bash
uv run poe test-model --model_id "your-username/text2sql-model-v1"
```

### Arguments

-   `--model_id`: The path to the local model directory (e.g., `outputs/checkpoint-500`) or a Hugging Face Model ID (e.g., `google/gemma-2b`).
-   `--dataset`: (Optional) Path to the test dataset. Defaults to `data/test_sft.jsonl`.
-   `--quantization`: (Optional) Load in 4-bit or 8-bit for faster inference on smaller GPUs.

### Example: Testing a Local Checkpoint

```bash
uv run poe test-model --model_id "outputs/qwen-text2sql/checkpoint-1000"
```

---

## 📂 Output Structure

Every evaluation run generates a permanent record in the `eval_results/` directory.

**Path Format:** `eval_results/{safe_model_id}/results_{timestamp}.json`

### JSON Content Example

The output file contains metadata about the run and a list of detailed results for each query.

```json
{
  "metadata": {
    "model_id": "outputs/qwen-text2sql",
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "test_dataset": "data/test_sft.jsonl",
    "timestamp": "2024-05-20T14:30:00",
    "accuracy": 85.5
  },
  "results": [
    {
      "question": "How many students are in the CS department?",
      "predicted_sql": "SELECT COUNT(*) FROM student WHERE dept_name = 'CS'",
      "gold_sql": "SELECT count(*) FROM student WHERE dept_name = 'CS'",
      "is_match": true,
      "latency_ms": 120
    },
    {
      "question": "List all course names.",
      "predicted_sql": "SELECT name FROM courses",
      "gold_sql": "SELECT title FROM courses",
      "is_match": false,
      "error": "Column 'name' not found in table 'courses'"
    }
  ]
}
```

---

## 📊 Evaluating Performance

The script automatically calculates **Execution Accuracy** (if a database connection is available) or **Exact Set Match Accuracy**.

### Metrics Explained
-   **Exact Match**: The predicted SQL matches the Gold SQL string exactly (ignoring case and whitespace).
-   **Execution Accuracy**: The predicted SQL returns the same result set from the database as the Gold SQL. This is the preferred metric as different queries can produce the same result.

---

## ⚖️ Benchmarking & Model Comparison

To compare two models (e.g., a baseline vs. a fine-tuned version):

1.  **Run Inference for Model A:**
    ```bash
    uv run poe test-model --model_id "google/gemma-2b"
    ```
2.  **Run Inference for Model B:**
    ```bash
    uv run poe test-model --model_id "outputs/my-fine-tuned-model"
    ```
3.  **Compare JSONs:**
    Check the `accuracy` score in the metadata of the generated JSON files.

### Improvement Workflow
1.  Identify queries where `is_match` is `false`.
2.  Analyze the `predicted_sql` to understand *why* it failed (e.g., wrong column name, hallucinated table).
3.  Add similar examples to your training data.
4.  Re-train and re-evaluate.

---

## 🧹 Artifact Management

The `eval_results/` directory can grow large.
-   **Git Ignore**: `eval_results/` is added to `.gitignore` to prevent bloating the repo.
-   **MLflow**: Use MLflow to log these JSON files as artifacts for long-term storage and easier comparison in the UI.
