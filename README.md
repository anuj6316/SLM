# SLM: Text-to-SQL Pipeline

A production-grade pipeline for fine-tuning Small Language Models (SLMs) on Text-to-SQL tasks. This project implements a full MLOps lifecycle, including automated data gathering, SQL cleaning, reverse-engineered schema anchoring, and Hugging Face integration.

## 🌟 Key Features

- **End-to-End Pipeline:** From raw data ingestion to model fine-tuning and evaluation.
- **Modular Architecture:** Easily swappable components for preprocessing, training, and inference.
- **Schema Anchoring:** Innovative technique to inject schema information into prompts, reducing hallucinations.
- **MLOps Best Practices:** Integrated with MLflow for experiment tracking and model registry.
- **Efficiency:** Optimized for fine-tuning SLMs (Small Language Models) using LoRA and QLoRA.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+**: The core programming language.
- **uv**: A fast Python package installer and resolver. [Install uv](https://github.com/astral-sh/uv).
- **Git**: Version control system.

## 🚀 Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/slm-text2sql.git
    cd slm-text2sql
    ```

2.  **Install Dependencies:**
    Use `uv` to sync the environment and install dependencies.
    ```bash
    uv sync
    ```

3.  **Environment Setup:**
    Create a `.env` file in the root directory and add your Hugging Face token.
    ```bash
    HF_TOKEN=your_hugging_face_token_here
    ```

## ⚡ Quick Start

Follow these steps to get your pipeline running in minutes.

### 1. Preprocess Data
Gather raw data, standardize SQL, and reverse-engineer database schemas.
```bash
uv run poe preprocess
```
*This command runs `text2sql_dataset_preprocessing/main.py`.*

### 2. Publish to Hugging Face
Upload the processed "Gold Standard" SFT dataset to your HF repository.
```bash
uv run poe publish
```
*This command runs `text2sql_dataset_preprocessing/src/publisher.py`.*

### 3. Train Model
Start the fine-tuning process using the configuration in `config.yaml`.
```bash
uv run python main.py train
```
*Note: Ensure your `config.yaml` is properly configured before training.*

### 4. Track Experiments
Launch the MLflow UI to monitor your training progress.
```bash
uv run mlflow ui --port 5000
```
Visit `http://localhost:5000` in your browser.

## ⚙️ Configuration

The pipeline is highly configurable via `config.yaml`. Key sections include:

-   **Data Ingestion**: Define source type (CSV, JSONL, HF), paths, and column mappings.
-   **Model Architecture**: Select the base model (e.g., Qwen, Llama), context window, and quantization.
-   **LoRA Settings**: Configure Rank, Alpha, and Target Modules for efficient fine-tuning.
-   **Training**: Set hyperparameters like learning rate, batch size, and epochs.
-   **Formatting**: Customize the system prompt and chat template.

See `config.yaml` for a detailed template.

## 📂 Project Structure

```
├── config.yaml                     # Main configuration file
├── data/                           # Data storage
├── docs/                           # Documentation and images
├── main.py                         # Entry point for training
├── pyproject.toml                  # Dependencies and tasks
├── README.md                       # Project documentation
├── text2sql_dataset_preprocessing/ # Data processing module
│   ├── main.py                     # Preprocessing entry point
│   └── src/                        # Processing logic
├── src/                            # Training and inference source code
│   ├── training/                   # Training scripts
│   └── inference/                  # Inference scripts (if applicable)
└── ...
```

## 📚 Documentation

For more detailed guides, check out:

-   **[SLM Guidebook](SLM_Guidebook.md)**: Deep dive into the "Gold Standard" SFT format, schema anchoring, and data cleaning.
-   **[MLflow Guide](MLFLOW_GUIDE.md)**: Learn how to track experiments, manage artifacts, and use the model registry.
-   **[Inference & Evaluation](INFERENCE_EVALUATION.md)**: Guide on running inference and evaluating model performance.

## 🤝 Contributing

We welcome contributions! Please fork the repository and submit a pull request. ensure your code passes all tests and linting checks.

## 📄 License

This project is licensed under the MIT License.
