"""
Schema Generator Module - Reverse-engineers database schemas from SQL queries.

Used as a fallback when external metadata files (tables.json) are unavailable.
"""
import json
import os
from typing import Dict, Set, List
import sqlglot
from sqlglot import exp
import logging
from tqdm import tqdm

logger = logging.getLogger("schema_generator")


def generate_schema_map(file_paths: List[str]) -> Dict[str, str]:
    """
    Reverse-engineers a schema map by scanning SQL queries in raw files.

    Args:
        file_paths: List of paths to JSONL files containing SQL queries

    Returns:
        Dict mapping db_id to schema string like "table1(col1, col2) | table2(col3)"

    Note:
        Uses heuristic column-table association. In JOIN queries, columns may be
        associated with multiple tables, which can lead to over-approximation.
    """
    internal_map: Dict[str, Dict[str, Set[str]]] = {}

    for path in file_paths:
        if not os.path.exists(path):
            continue

        logger.info(f"Extracting schemas from {os.path.basename(path)}...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Scanning for schema"):
                try:
                    entry = json.loads(line)
                    db_id = entry.get("db_id", "default_db")

                    sql = entry.get("query") or entry.get("sql") or entry.get("SQL")
                    if not sql or not isinstance(sql, str):
                        continue

                    parsed = sqlglot.parse_one(sql, read="sqlite")

                    if db_id not in internal_map:
                        internal_map[db_id] = {}

                    tables_in_query: List[str] = []
                    for table in parsed.find_all(exp.Table):
                        t_name = table.name.lower()
                        tables_in_query.append(t_name)
                        if t_name not in internal_map[db_id]:
                            internal_map[db_id][t_name] = set()

                    for column in parsed.find_all(exp.Column):
                        col_name = column.name.lower()
                        if col_name == "*":
                            continue

                        for t in tables_in_query:
                            internal_map[db_id][t].add(col_name)

                except Exception:
                    continue

    final_map: Dict[str, str] = {}
    for db_id, tables in internal_map.items():
        parts = []
        for t_name, cols in tables.items():
            col_str = ", ".join(sorted(list(cols)))
            parts.append(f"{t_name}({col_str})")
        final_map[db_id] = " | ".join(parts)

    logger.info(f"Generated {len(final_map)} database schemas from SQL analysis")
    return final_map
