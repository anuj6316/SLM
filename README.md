# Text-to-SQL Fine-Tuning Pipeline (Unsloth + SageMaker)

This repository contains a modular, production-grade pipeline for fine-tuning Large Language Models (LLMs) on Text-to-SQL tasks. It is optimized for the **Spider** dataset but is flexible enough to handle any instruction-based dataset.

The project supports two modes of operation:
1.  **Local Development:** Run everything on your own machine (requires GPU).
2.  **Cloud Training:** Seamlessly scale training to AWS SageMaker Spot Instances.

## 📁 Project Structure

```text
SLM/
├── main.py                 # CLI Orchestrator (The main entry point)
├── run_sagemaker.py        # Script to launch AWS Training Jobs
├── src/
│   ├── train.py            # Generic SFT training script (Unsloth)
│   ├── inference.py        # Robust inference with Schema Injection
│   ├── schema_utils.py     # Database schema handling logic
│   ├── requirements.txt    # Dependencies for the Cloud Container
│   └── ...
└── data/                   # Your datasets (Spider, etc.)
```

---

## 🚀 Quick Start (Local)

### 1. Setup Environment
```bash
# Install dependencies locally
pip install -r src/requirements.txt
pip install sagemaker boto3  # If you plan to use AWS
```

### 2. The Orchestrator (`main.py`)
All local tasks are handled by `main.py`.

**Step A: Prepare Data**
Converts the raw Spider dataset into a schema-augmented `.jsonl` format.
```bash
python main.py prepare
```

**Step B: Train the Model**
Fine-tunes Qwen-2.5-1.5B (or any model) on the prepared data.
```bash
# Default: 2 epochs
python main.py train --epochs 2 --base_model "Qwen/Qwen2.5-1.5B-Instruct"
```

**Step C: Run Inference**
Generates SQL queries for a test set (e.g., `dev.json`).
```bash
python main.py inference --inference_data data/spider/dev.json --inference_output my_results.json
```

**Step D: Evaluate**
Runs the official Spider evaluation script to get Execution Accuracy.
```bash
python main.py evaluate
```

---

## ☁️ Running on AWS SageMaker

This project is "Cloud-Ready." You can move your training to a powerful GPU in the cloud without changing your code.

### Prerequisites
1.  **AWS Credentials:** Configure your local machine with `aws configure`.
2.  **S3 Bucket:** Upload your `data/spider_sft` folder to an S3 bucket (e.g., `s3://my-bucket/data/`).
3.  **IAM Role:** Ensure you have an IAM Role ARN with `AmazonSageMakerFullAccess` and `AmazonS3FullAccess`.

### How to Launch
1.  Open `run_sagemaker.py`.
2.  Update the **Configuration** section:
    ```python
    S3_BUCKET = "your-bucket-name"
    ROLE_ARN = "arn:aws:iam::..."
    ```
3.  Run the script:
    ```bash
    python run_sagemaker.py
    ```

### Why SageMaker?
*   **Spot Instances:** This script is configured to use Spot Instances (`use_spot_instances=True`), saving you up to 70% on compute costs.
*   **Zero Setup:** SageMaker automatically builds the environment using `src/requirements.txt`.
*   **Auto-Termination:** The GPU machine shuts down immediately after training finishes.

---

## 🛠️ Advanced Customization

### Using Custom Datasets
You can use this pipeline for **non-SQL** tasks too.
1.  Create a `.jsonl` file with `{"instruction": "...", "output": "..."}` format.
2.  Upload it to S3.
3.  Point `run_sagemaker.py` to that S3 location.
4.  The `train.py` script will automatically detect and train on it.

### Changing the Base Model
To fine-tune Llama-3, Mistral, or others, simply change the argument:
```bash
python main.py train --base_model "unsloth/llama-3-8b-bnb-4bit"
```

---

## 🐞 Troubleshooting

**"ModuleNotFoundError: No module named 'datasets'" on SageMaker**
*   Ensure `src/requirements.txt` exists and contains `datasets`. SageMaker installs from this file *inside* the container.

**Empty SQL Output ("SQL:")**
*   This usually means the model didn't receive the Database Schema in the prompt. Ensure you are using the correct `inference.py` flow which uses `schema_utils.py`.

**SageMaker Job Fails Immediately**
*   Check CloudWatch logs. Common issue: The IAM Role doesn't have permission to read the S3 bucket.