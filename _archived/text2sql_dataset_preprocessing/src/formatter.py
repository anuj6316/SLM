"""
SFT Formatter Module - Converts cleaned records to instruction-following format.

Output format: {"instruction": ..., "input": ..., "output": ..., "metadata": ...}
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("formatter")

FALLBACK_SCHEMA_PREFIXES = ("Schema not available", "Schema context not found", "Schema context not available")


def normalize_to_sft(
    entry: Dict[str, Any],
    dataset_type: str,
    schema_map: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Converts cleaned records into a standard instruction-following format.

    Args:
        entry: Cleaned dataset record with 'question' and 'cleaned_sql'
        dataset_type: One of 'spider', 'bird', 'gretel'
        schema_map: Mapping of db_id to schema string

    Returns:
        SFT-formatted dict, or None if required fields missing
    """
    schema_text = ""
    if dataset_type in ["spider", "bird"]:
        db_id = entry.get("db_id", "unknown")
        schema_text = schema_map.get(db_id, "Schema not available.")
        if db_id == "unknown" or db_id not in schema_map:
            logger.warning(f"Missing schema for db_id='{db_id}' in {dataset_type} dataset")
    elif dataset_type == "gretel":
        schema_text = entry.get("context", entry.get("sql_context", "Schema context not found."))
    else:
        schema_text = "Schema context not available."

    if schema_text.startswith(FALLBACK_SCHEMA_PREFIXES):
        logger.debug(f"Using fallback schema text for entry")

    question = entry.get("question", "")
    sql = entry.get("cleaned_sql", "")

    if not sql or not question:
        return None

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
