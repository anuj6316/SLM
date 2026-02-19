# MLflow Integration Guide: Text-to-SQL MLOps

This guide explains how to use **MLflow** as the "Laboratory Notebook" for your Text-to-SQL project. It will track your data preprocessing, log your training experiments, and manage your model versions.

---

## 🏗 What is MLflow?
In a production MLOps circle, MLflow serves three main purposes:
1. **Tracking:** Records every run (code used, parameters, and results).
2. **Artifacts:** Stores the actual files produced (cleaned data, LoRA adapters).
3. **Registry:** Acts as a gatekeeper to tag models as "Staging" or "Production."

---

## 🚀 Step 1: Installation & Setup

### 1. Add MLflow to your project
```bash
uv add mlflow
```

### 2. Launch the Dashboard
Open a separate terminal and run:
```bash
uv run mlflow ui --port 5000
```
Visit `http://localhost:5000` in your browser. This is where you will see your experiments.

---

## 🔄 Step 2: Tracking Data Preprocessing
We want to track the "Data DNA"—exactly how many rows were cleaned and what the final file looked like.

**In `text2sql_dataset_preprocessing/main.py`:**
1. **Set Experiment:** `mlflow.set_experiment("Text2SQL_Data_Ops")`
2. **Start Run:** Wrap the processing loop in `with mlflow.start_run():`.
3. **Log Stats:** Use `mlflow.log_param("raw_files", len(raw_files))` and `mlflow.log_metric("final_row_count", processed_count)`.
4. **Save Artifact:** Use `mlflow.log_artifact("data/train_sft.jsonl")`.

---

## 🧠 Step 3: Tracking Model Training
Since you are using `SFTTrainer` (Hugging Face), integration is nearly automatic.

**In `src/training/train.py`:**
1. **Enable Logging:** In your `SFTConfig`, change `report_to="none"` to `report_to="mlflow"`.
2. **Set Experiment Name:**
   ```bash
   export MLFLOW_EXPERIMENT_NAME="Text2SQL_Training"
   ```
3. **Auto-Tracking:** MLflow will now automatically capture:
   - **Hyperparameters:** `learning_rate`, `lora_r`, `lora_alpha`.
   - **Metrics:** `train_loss`, `eval_loss` (every 50 steps).
   - **System:** GPU memory usage and CPU power.

---

## 🏆 Step 4: The Model Registry (Champion vs Challenger)
After training 3 different models (e.g., Gemma-2b, Qwen-1.5b), you compare them in the MLflow UI.

1. **Register:** Click on your best run and select "Register Model." Name it `Text2SQL_SLM`.
2. **Tagging:** 
   - **None:** A fresh experiment.
   - **Staging:** Passed initial loss checks, ready for "Execution Accuracy" testing.
   - **Production:** Passed all tests and is currently serving the API.

---

## 🤖 Step 5: Professional Automation
Add these tasks to your `pyproject.toml` to make MLOps effortless:

```toml
[tool.poe.tasks]
mlflow-ui = "mlflow ui --port 5000"
# Run the pipeline and automatically log everything to MLflow
train-tracked = "python main.py train --config config.yaml" 
```

---

## 📈 The Professional MLOps Workflow
1. **Develop:** Modify your `cleaner.py` to handle new edge cases.
2. **Preprocess:** Run `poe preprocess`. Check MLflow to see if the row count improved.
3. **Train:** Run `poe train-tracked`. Watch the loss curves live in the dashboard.
4. **Evaluate:** Run your `evaluate.py` script. It will update the MLflow run with **Execution Accuracy**.
5. **Promote:** If Accuracy > 85%, tag that model as **Production** in the registry.
6. **Serve:** Your vLLM server pulls the `Production` tag and updates the live API.
