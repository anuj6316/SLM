# MLflow Integration Guide: Text-to-SQL MLOps

This guide explains how to use **MLflow** as the "Laboratory Notebook" for your Text-to-SQL project. It allows you to track experiments, compare model performance, and manage model versions from staging to production.

---

## 🏗 What is MLflow?

In our production pipeline, MLflow serves three main purposes:

1.  **Tracking**: Records every run, including code versions, hyperparameters, and metrics.
2.  **Artifacts**: Stores output files (cleaned datasets, model checkpoints, logs).
3.  **Registry**: Acts as a gatekeeper to tag models as "Staging" or "Production."

---

## 📋 Prerequisites

Ensure MLflow is installed in your environment. If you followed the main installation guide, it should already be present.

To verify:
```bash
uv run mlflow --version
```

If not installed:
```bash
uv add mlflow
```

---

## 🚀 Getting Started

To view your experiments, you need to launch the MLflow Tracking Server.

1.  **Start the Server:**
    ```bash
    uv run mlflow ui --port 5000
    ```

2.  **Access the Dashboard:**
    Open your browser and navigate to [http://localhost:5000](http://localhost:5000).

---

## 🔍 Tracking Experiments

The pipeline is pre-configured to log runs automatically.

### 1. Data Preprocessing
When you run `uv run poe preprocess`, the pipeline logs:
-   **Parameters**: Number of raw files processed.
-   **Metrics**: Final row count, number of dropped records.
-   **Artifacts**: The generated `train_sft.jsonl` file.

**Experiment Name**: `Text2SQL_Data_Ops`

### 2. Model Training
When you run `uv run python main.py train`, the pipeline uses Hugging Face's `SFTTrainer` integration to log:
-   **Hyperparameters**: Learning rate, batch size, LoRA rank (`r`), LoRA alpha, etc.
-   **Metrics**: Training loss, evaluation loss (logged every `logging_steps`).
-   **System Metrics**: GPU memory usage, CPU utilization.

**Experiment Name**: `Text2SQL_Training`

---

## 🏆 Model Registry (Champion vs Challenger)

The Model Registry allows you to manage the lifecycle of your models.

1.  **Compare Runs**: In the MLflow UI, select multiple runs to compare their loss curves.
2.  **Register a Model**:
    -   Click on the best-performing run.
    -   Click the "Register Model" button.
    -   Name it `Text2SQL_SLM` (or your preferred name).
3.  **Transition Stages**:
    -   **Staging**: The model is ready for testing.
    -   **Production**: The model has passed all tests and is ready for deployment.
    -   **Archived**: Old models that are no longer in use.

---

## 📊 Visualizing Results

In the MLflow UI:

-   **Parallel Coordinates Plot**: Great for seeing which hyperparameters (e.g., `learning_rate`, `lora_r`) lead to the lowest loss.
-   **Scatter Plot**: Visualize the relationship between two metrics (e.g., Training Steps vs. Loss).
-   **Artifact View**: Inspect the actual files (config, logs) associated with a run.

---

## ❓ Troubleshooting

**Q: I don't see my runs in the UI.**
A: Ensure you are running the `mlflow ui` command from the root of the project where the `mlruns` directory is located.

**Q: The experiment name is "Default".**
A: Set the `MLFLOW_EXPERIMENT_NAME` environment variable before running your script:
```bash
export MLFLOW_EXPERIMENT_NAME="Text2SQL_Training"
```
Or ensure it's set in your code via `mlflow.set_experiment()`.

**Q: "Connection Refused" error.**
A: Make sure no other service is using port 5000. You can change the port:
```bash
uv run mlflow ui --port 5001
```
