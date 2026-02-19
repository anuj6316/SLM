# Model-Centric Inference & Evaluation Guide

This document outlines the approach for documenting and comparing Text-to-SQL model outputs based on specific **Model IDs**. This is a core requirement for MLOps to ensure traceability and prevent performance regressions.

---

## 🏗 Concept: The "Evaluation Evidence"
Instead of overwriting a single `results.json`, we treat every evaluation as a permanent record.
- **Location:** `eval_results/{safe_model_id}/`
- **Filename:** `results_{timestamp}.json`
- **Content:** The model's predictions, the gold standard SQL, and the exact prompt context used.

---

## 🚀 Implementation Strategy

### 1. The Inference Logic
The inference script should be modified to:
1. Accept a `--model_id` (e.g., `anuj6316/text2sql-v1` or `./outputs/v2`).
2. Create a "Safe Name" for the model (e.g., replace `/` with `_`).
3. Generate SQL for each question in your test set.
4. Save a JSON blob that includes both the **Predictions** and the **Model Metadata**.

### 2. Result Structure (Example)
Every inference run should produce a JSON file structured like this:
```json
{
  "metadata": {
    "model_id": "anuj6316/text2sql-gemma-2b",
    "base_model": "google/gemma-2b",
    "test_dataset": "data/test_sft.jsonl",
    "timestamp": "2026-02-19T17:45:00"
  },
  "results": [
    {
      "db_id": "college_1",
      "question": "How many students are in the CS department?",
      "predicted_sql": "SELECT COUNT(*) FROM student WHERE dept_name = 'CS'",
      "gold_sql": "SELECT count(*) FROM student WHERE dept_name = 'CS'",
      "is_match": true
    }
  ]
}
```

---

## 🛠 Automation with Poe
Add this task to your `pyproject.toml` to standardize the evaluation:

```toml
[tool.poe.tasks]
# Example: uv run poe test-model --model_id "anuj6316/text2sql-v1"
test-model = "python src/training/inference.py"
```

---

## 📈 MLOps Benchmarking Workflow
1. **Inference:** Run the script for `Model_A` and `Model_B`.
2. **Comparison:** Use a script or the MLflow UI to compare the generated JSON files in `eval_results/`.
3. **Identification:** Find queries where `Model_A` succeeded but `Model_B` failed.
4. **Iterate:** Add those failing queries to your training set for the next version.

---

## 🚿 Clean-up Requirement
Remember to add `eval_results/` to your `.gitignore` to keep your code repository clean, while ensuring these results are logged as **Artifacts** in MLflow or Hugging Face.
