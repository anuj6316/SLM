# SLM Guidebook: Text-to-SQL Success

![Text-to-SQL MLOps Pipeline Flow](https://storage.googleapis.com/second-petal-295822.appspot.com/elements/elements%3Ad6a8a796d7c03b962ceaf006140dbc0218daf65f7ac9c3b5595641b2c787654b.png)

This guide documents the technical decisions and methodologies used to achieve high accuracy in Text-to-SQL tasks with Small Language Models.

## 🔑 The "Gold Standard" SFT Format
SLMs (under 7B parameters) require high-quality instruction tuning. Every record in our `train_sft.jsonl` follows this structure:

- **Instruction:** Task definition (Standardized).
- **Input:** "### Database Schema: {schema} ### Question: {question}".
- **Output:** Canonicalized SQL (One-lined, consistent casing).

## 🧠 Reverse-Engineered Schema Anchoring
When external metadata (`tables.json`) is missing, our pipeline triggers a **Schema Generator**.
1. **Scan:** Iterates through all SQL queries in the raw dataset.
2. **Extract:** Uses `sqlglot` to find every table and column mentioned in the queries.
3. **Map:** Associates columns with tables to build a "Virtual Schema."
4. **Anchor:** Injecting this virtual schema into the prompt prevents model hallucinations and "Schema not available" errors.

## 🚿 Data Cleaning Methodology
We use `sqlglot` for **Syntactic Validation** and **Canonicalization**.
- **Normalization:** Converts varied SQL styles into a uniform SQLite dialect.
- **Filtering:** Automatically drops records with broken SQL syntax to ensure the model never learns "bad" data.
- **Whitespace Sanitization:** Strips natural language questions of extra spaces and hidden characters.

## 📈 MLOps Circle
1. **GitHub:** Stores only logic and configuration.
2. **HF Hub:** Acts as the Model and Data Registry.
3. **Poe:** Standardizes execution across environments.
