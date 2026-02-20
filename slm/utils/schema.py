"""
Schema loading utilities for Text-to-SQL.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def load_schema_dict(tables_path: Path) -> Dict[str, str]:
    """
    Load schemas from Spider/BIRD style tables.json.

    Args:
        tables_path: Path to the tables.json file

    Returns:
        Dict mapping db_id to schema string like "table1(col1, col2) | table2(col3)"
    """
    if not tables_path.exists():
        logger.warning(f"Schema file not found: {tables_path}")
        return {}

    with open(tables_path, "r") as f:
        tables_data: List[Dict[str, Any]] = json.load(f)

    schema_map: Dict[str, str] = {}
    for db in tables_data:
        db_id = db["db_id"]
        schema_map[db_id] = _parse_db_schema(db)

    logger.info(f"Loaded {len(schema_map)} database schemas")
    return schema_map


def _parse_db_schema(db: Dict[str, Any]) -> str:
    """Parse a single database entry into schema string."""
    table_names = db.get("table_names_original", [])
    column_names = db.get("column_names_original", [])

    tables_dict: Dict[str, List[str]] = {}
    for table_idx, col_name in column_names:
        if table_idx == -1:
            continue
        t_name = table_names[table_idx]
        tables_dict.setdefault(t_name, []).append(col_name)

    parts = [f"{t}({', '.join(cols)})" for t, cols in tables_dict.items()]
    return " | ".join(parts)
