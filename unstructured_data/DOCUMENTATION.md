# Unstructured Data to QA Dataset Pipeline

This project is a high-performance pipeline designed to transform unstructured web content into high-quality, structured Question-Answer (QA) datasets. It leverages **Jina.ai** for web scraping and **Groq Cloud (LLMs)** for intelligent QA generation and self-judging.

---

## 🏗️ Architecture Overview

The system operates in two primary phases, organized into a 5-layer pipeline:

1.  **Input Layer:** Configuration and URL ingestion.
2.  **Scraping Stage:** Multi-strategy scraping (Flash/Deep) using Jina.ai.
3.  **Refinement Stage:** Regex-based cleaning of raw markdown.
4.  **AI Generation Stage:** Chunking, Question/Answer generation, and Quality Judging.
5.  **Output Layer:** Hashed and metadata-enriched JSONL storage.

> **Note:** A detailed technical diagram can be found in `docs/diagrams/technical_flow.eraser`.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- [Jina.ai API Key](https://jina.ai/reader/)
- [Groq API Key](https://console.groq.com/)

### 2. Installation
```bash
# Clone the repository
git clone <repository-url>
cd unstructured_data

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```env
JINA_API_KEY=your_jina_key_here
GROQ_API_KEY=your_groq_key_here
```

---

## ⚙️ Configuration

The pipeline is controlled via `config.yml`.

| Section | Key | Description |
| :--- | :--- | :--- |
| `web_scrapping` | `url` | The target website to scrape. |
| | `scrape_type` | `flash` (single page) or `deep` (recursive domain crawl). |
| | `remove_temp_files` | Automatically delete intermediate markdown files. |
| `ProcessMarkdownQAPairs` | `model_id` | The LLM to use (e.g., `llama-3.1-70b-versatile`). |
| | `api_key` | Path to the Groq API key (uses `${GROQ_API_KEY}` by default). |

---

## 🛠️ Core Components

### 1. Web Scraper (`scrape_flow.py`)
- **Flash Scrape:** Fetches the target URL as clean markdown in one pass.
- **Deep Scrape:** Uses a recursive `deque` queue to crawl the entire domain.
- **Cleaning:** Strips images, SVGs, and unnecessary URLs to minimize token usage.

### 2. QA Generator (`qa_generator.py`)
- **Chunking:** Uses `RecursiveCharacterTextSplitter` from LangChain to handle large documents.
- **LLM Logic:**
    - **Question Gen:** Creates diverse questions based on chunk content.
    - **Answer Gen:** Generates precise answers for each question.
    - **Judge Review:** An independent LLM call scores the QA pair (0-10) and provides reasoning.
- **Safe Invoke:** Implements exponential backoff to handle Rate Limits (429 errors) gracefully.

### 3. Utilities (`utils.py`)
- **Hashing:** Every chunk is uniquely hashed (SHA-256) to ensure data integrity and prevent duplicates.
- **Rich UI:** Provides beautiful console tables for configuration display.

---

## 🖥️ Usage

### CLI Mode (Standard)
Run the full pipeline from start to finish:
```bash
python main.py
```

### UI Mode (Gradio)
Launch a web-based prototype to test scraping logic:
```bash
python app.py
```

---

## 📈 Recent Improvements

- **Sticky Progress Bar:** Switched all internal `print()` calls to `tqdm.write()` to ensure the progress bar remains at the bottom of the terminal during long runs.
- **Silent Logging:** Suppressed verbose `httpx` and `openai` network logs to focus only on application-level status and errors.
- **Parsing Robustness:** Improved Pydantic output parsing with better error handling and raw output logging for debugging.

---

## 📂 Directory Structure
- `pipeline/`: ZenML pipeline definitions (Optional).
- `prompts/`: System and user prompt templates for LLM logic.
- `steps/`: Individual processing steps for modular workflows.
- `docs/diagrams/`: Eraser DSL files for technical visualization.
- `final_output.jsonl`: The resulting dataset (Auto-generated).
