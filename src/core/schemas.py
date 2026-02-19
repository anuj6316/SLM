import os
import json
import re
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

# Use standard logging for consistency with inference.py
logger = logging.getLogger(__name__)

def load_schema_dict(tables_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads database schemas from tables.json.
    
    Returns a dictionary mapping db_id to its schema information:
    {
        db_id: {
            "tables": {table_name: [columns]},
            "foreign_keys": [(table1, col1, table2, col2)],
            "graph": {table_name: {connected_tables}}
        }
    }
    """
    logger.info(f"Loading schemas from: {tables_path}")
    if not os.path.exists(tables_path):
        logger.error(f"Schema file not found: {tables_path}")
        raise FileNotFoundError(tables_path)

    try:
        with open(tables_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse schema JSON: {str(e)}")
        raise

    db_schemas = {}

    for db in data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        column_names = db["column_names_original"]
        foreign_keys = db["foreign_keys"]

        tables = defaultdict(list)
        # Build table -> columns mapping
        for table_idx, col_name in column_names:
            if table_idx == -1: # Skip special index
                continue
            tables[table_names[table_idx]].append(col_name)

        # Build Foreign Key relations and connection graph
        fk_list = []
        graph = defaultdict(set)

        for src, tgt in foreign_keys:
            src_table_idx, src_col = column_names[src]
            tgt_table_idx, tgt_col = column_names[tgt]

            t1 = table_names[src_table_idx]
            t2 = table_names[tgt_table_idx]

            fk_list.append((t1, src_col, t2, tgt_col))
            graph[t1].add(t2)
            graph[t2].add(t1)

        db_schemas[db_id] = {
            "tables": dict(tables),
            "foreign_keys": fk_list,
            "graph": {k: list(v) for k, v in graph.items()},
        }

    return db_schemas

def tokenize(text: str) -> Set[str]:
    """Simple alphanumeric tokenizer."""
    return set(re.findall(r"\w+", text.lower()))

def find_relevant_tables(question: str, schema: Dict[str, Any], db_id: str = "") -> Set[str]:
    """
    Identifies tables that are likely relevant to the question based on keyword matching
    with table names and column names.
    """
    tokens = tokenize(question)
    matched = set()

    for table, cols in schema["tables"].items():
        # Match table name
        if table.lower() in tokens:
            matched.add(table)
            continue

        # Match column names
        for col in cols:
            if col.lower() in tokens:
                matched.add(table)
                break

    if not matched:
        logger.debug(f"[{db_id}] No tables matched. Defaulting to all tables.")
        return set(schema["tables"].keys())

    return matched

def expand_with_foreign_keys(selected_tables: Set[str], schema: Dict[str, Any], max_hops: int = 1, db_id: str = "") -> Set[str]:
    """
    Expands the set of selected tables by including their neighbors in the 
    Foreign Key graph to ensure join paths are available.
    """
    graph = schema["graph"]
    expanded = set(selected_tables)

    for _ in range(max_hops):
        new_tables = set()
        for table in expanded:
            neighbors = graph.get(table, [])
            new_tables.update(neighbors)
        
        if new_tables.issubset(expanded):
            break
        expanded.update(new_tables)

    return expanded

def format_schema(db_id: str, schema: Dict[str, Any], selected_tables: Set[str]) -> str:
    """Formats the schema into a human-readable string for the prompt."""
    lines = [f"Database: {db_id}", "Tables:"]

    for table in sorted(selected_tables):
        cols = schema["tables"][table]
        cols_str = ", ".join(cols)
        lines.append(f"{table}({cols_str})")

    # Filter foreign keys to only include those between selected tables
    fk_lines = []
    fk_lines.extend(
        f"{t1}.{c1} -> {t2}.{c2}"
        for t1, c1, t2, c2 in schema["foreign_keys"]
        if t1 in selected_tables and t2 in selected_tables
    )
    if fk_lines:
        lines.append("Foreign Keys:")
        lines.extend(fk_lines)

    return "\n".join(lines)

def build_sft_prompt(db_id: str, schema: Dict[str, Any], question: str) -> str:
    """
    Centralized prompt builder. 
    Matches tables, expands via FKs, and formats the final prompt text.
    """
    matched = find_relevant_tables(question, schema, db_id)
    selected_tables = expand_with_foreign_keys(matched, schema, db_id=db_id)
    schema_text = format_schema(db_id, schema, selected_tables)
    
    return f"{schema_text}\n\nQuestion: {question}\nSQL:"