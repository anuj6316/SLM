import json
import os
import sqlglot
from sqlglot import exp
import logging
from tqdm import tqdm

logger = logging.getLogger("schema_generator")

def generate_schema_map(file_paths: list):
    """
    Reverse-engineers a schema map by scanning SQL queries in raw files.
    Returns: {db_id: "table1(col1, col2) | table2(col3)"}
    """
    # db_id -> table_name -> set(columns)
    internal_map = {}

    for path in file_paths:
        if not os.path.exists(path):
            continue
        
        logger.info(f"Extracting schemas from {os.path.basename(path)}...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Scanning for schema"):
                try:
                    entry = json.loads(line)
                    db_id = entry.get("db_id", "default_db")
                    
                    # Try common SQL keys used in Spider, BIRD, and Gretel
                    sql = entry.get("query") or entry.get("sql") or entry.get("SQL")
                    if not sql or not isinstance(sql, str):
                        continue

                    # Parse SQL to find used tables and columns
                    parsed = sqlglot.parse_one(sql, read="sqlite")
                    
                    if db_id not in internal_map:
                        internal_map[db_id] = {}

                    # Extract all tables
                    tables_in_query = []
                    for table in parsed.find_all(exp.Table):
                        t_name = table.name.lower()
                        tables_in_query.append(t_name)
                        if t_name not in internal_map[db_id]:
                            internal_map[db_id][t_name] = set()

                    # Extract all columns
                    for column in parsed.find_all(exp.Column):
                        col_name = column.name.lower()
                        if col_name == "*": continue
                        
                        # Heuristic: Associate columns with all tables in this specific query
                        for t in tables_in_query:
                            internal_map[db_id][t].add(col_name)
                            
                except Exception:
                    # Skip malformed SQL or JSON lines
                    continue

    # Format the collected sets into the string format expected by the pipeline
    final_map = {}
    for db_id, tables in internal_map.items():
        parts = []
        for t_name, cols in tables.items():
            col_str = ", ".join(sorted(list(cols)))
            parts.append(f"{t_name}({col_str})")
        final_map[db_id] = " | ".join(parts)
    
    return final_map
