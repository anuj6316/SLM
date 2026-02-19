# SLM Guidebook: Text-to-SQL Success

![Text-to-SQL MLOps Pipeline Flow](https://storage.googleapis.com/second-petal-295822.appspot.com/elements/elements%3Ad6a8a796d7c03b962ceaf006140dbc0218daf65f7ac9c3b5595641b2c787654b.png)

This guide documents the technical decisions, data formats, and methodologies used to achieve high accuracy in Text-to-SQL tasks with Small Language Models (SLMs).

## 🔑 The "Gold Standard" SFT Format

SLMs (under 7B parameters) require high-quality, consistent instruction tuning. We transform all diverse datasets into a unified JSONL format.

### Structure

Every record in our `train_sft.jsonl` follows this strict structure:

-   **Instruction**: A standardized task definition.
-   **Input**: A structured prompt containing the database schema and the user's question.
-   **Output**: The canonicalized SQL query (one-lined, consistent casing).

### Example Record

```json
{
  "instruction": "You are a text-to-SQL expert. Given the database schema and a question, generate the correct SQL query.",
  "input": "### Database Schema: CREATE TABLE students (id INTEGER, name TEXT, age INTEGER, major TEXT); ### Question: Show me the names of all students majoring in Computer Science.",
  "output": "SELECT name FROM students WHERE major = 'Computer Science'"
}
```

## 🚿 Data Pipeline & Transformation

The pipeline transforms raw data into the Gold Standard format through several stages:

1.  **Ingestion**: Raw data is loaded from CSVs, JSON, or Hugging Face datasets.
2.  **Normalization**:
    -   Column names are standardized.
    -   SQL dialects are converted to SQLite (or target dialect) using `sqlglot`.
3.  **Schema Injection**: If schema metadata is missing, we use **Reverse-Engineered Schema Anchoring**.
4.  **Validation**:
    -   SQL queries are parsed to ensure syntactic correctness.
    -   Broken queries are automatically filtered out.
5.  **Formatting**: The final JSONL is generated, ready for training.

## 🧠 Reverse-Engineered Schema Anchoring

One of the biggest challenges in Text-to-SQL is dealing with datasets that lack explicit schema definitions (`CREATE TABLE` statements). Our pipeline solves this by "reverse-engineering" the schema from the SQL queries themselves.

### How it Works

1.  **Scan**: The pipeline iterates through all SQL queries in the raw dataset.
2.  **Extract**: It uses `sqlglot` to parse each query and identify every table and column mentioned.
3.  **Map**: It builds a "Virtual Schema" map, associating columns with their respective tables.
4.  **Anchor**: This virtual schema is formatted as a `CREATE TABLE` string and injected into the prompt.

### Example

**Raw Query:**
```sql
SELECT t.name, c.course_title
FROM teachers t
JOIN courses c ON t.id = c.teacher_id
WHERE t.department = 'Science'
```

**Extracted Virtual Schema:**
```sql
CREATE TABLE teachers (id INTEGER, name TEXT, department TEXT);
CREATE TABLE courses (teacher_id INTEGER, course_title TEXT);
```

**Result:** The model receives this context, preventing hallucinations about non-existent columns.

## 🧹 Data Cleaning Methodology

We rely heavily on `sqlglot` for ensuring data quality.

-   **Syntactic Validation**: If `sqlglot.transpile(sql)` fails, the record is dropped. This ensures the model never learns from broken SQL.
-   **Canonicalization**:
    -   Keywords are capitalized (`select` -> `SELECT`).
    -   Indentation is removed for token efficiency.
    -   Quote styles are standardized.
-   **Whitespace Sanitization**: Natural language questions are stripped of double spaces, tabs, and hidden characters.

## ➕ Extending the Pipeline

To add a new dataset:

1.  Add the source file to `data/`.
2.  Update `config.yaml` with the new file path and column mappings.
    ```yaml
    data:
      source_type: "csv"
      path: "data/new_dataset.csv"
      columns:
        instruction: "question_col"
        input: ["col1", "col2"] # Columns to join for context
        output: "sql_col"
    ```
3.  Run `uv run poe preprocess` to regenerate the training data.

## 📈 MLOps Circle

1.  **GitHub**: Stores source code, configuration, and documentation.
2.  **Hugging Face Hub**: Acts as the central registry for:
    -   **Datasets**: Versioned `train_sft.jsonl` files.
    -   **Models**: Fine-tuned LoRA adapters and merged models.
3.  **Poe**: A task runner that standardizes commands across local and CI/CD environments.
