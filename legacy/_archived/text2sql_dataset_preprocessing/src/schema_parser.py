"""
Schema Parser Module - Parses Spider/BIRD style tables.json files.

Extracts database schema metadata and creates a mapping of db_id to schema strings.
"""
import json
import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger("schema_parser")


def load_spider_schemas(tables_json_path: str) -> Dict[str, str]:
    """
    Parses Spider/BIRD style tables.json to create a mapping of {db_id: schema_string}.

    Args:
        tables_json_path: Path to the tables.json file

    Returns:
        Dict mapping db_id to schema string like "table1(col1, col2) | table2(col3)"
    """
    if not os.path.exists(tables_json_path):
        logger.warning(f"Schema file not found at {tables_json_path}. Proceeding with placeholders.")
        return {}

    try:
        with open(tables_json_path, 'r', encoding='utf-8') as f:
            tables_data: List[Dict[str, Any]] = json.load(f)

        schema_map: Dict[str, str] = {}
        for db in tables_data:
            db_id = db["db_id"]
            table_names = db.get("table_names_original", [])
            column_names = db.get("column_names_original", [])

            tables_dict: Dict[str, List[str]] = {}
            for table_idx, col_name in column_names:
                if table_idx == -1:
                    continue
                t_name = table_names[table_idx]
                tables_dict.setdefault(t_name, []).append(col_name)

            schema_parts = [f"{t}({', '.join(cols)})" for t, cols in tables_dict.items()]
            schema_map[db_id] = " | ".join(schema_parts)

        logger.info(f"Loaded {len(schema_map)} database schemas from {tables_json_path}")
        return schema_map
    except Exception as e:
        logger.error(f"Error parsing schema file: {e}")
        return {}
