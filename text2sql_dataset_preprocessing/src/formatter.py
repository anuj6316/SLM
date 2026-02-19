import logging

def normalize_to_sft(entry: dict, dataset_type: str, schema_map: dict):
    """
    Converts cleaned records into a standard instruction-following format.
    Goal: High accuracy through explicit Schema Context + Cleaned SQL.
    """
    # 1. Get Schema Context
    schema_text = ""
    if dataset_type in ["spider", "bird"]:
        db_id = entry.get("db_id", "unknown")
        schema_text = schema_map.get(db_id, "Schema not available.")
    elif dataset_type == "gretel":
        schema_text = entry.get("context", entry.get("sql_context", "Schema context not found."))
    else:
        schema_text = "Schema context not available."

    # 2. Extract Cleaned Fields
    question = entry.get("question", "")
    sql = entry.get("cleaned_sql", "")

    if not sql or not question:
        return None

    # 3. Create the SFT Structure
    # Using triple quotes to handle multi-line f-strings safely
    return {
        "instruction": "Convert the following natural language question into a valid SQL query based on the provided database schema.",
        "input": f"""### Database Schema:
{schema_text}

### Question:
{question}""",
        "output": sql,
        "metadata": {
            "dataset": dataset_type,
            "db_id": entry.get("db_id", "none")
        }
    }
