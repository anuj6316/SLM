import json
from collections import defaultdict
import re
from src.utils import get_logger

logger = get_logger(__name__)

def load_schema_dict(tables_path):
    """
    Returns:
    {
        db_id: {
            "tables": {table_name: [columns]},
            "foreign_keys": [(table1, col1, table2, col2)],
            "graph": adjacency list for table joins
        }
    }
    """
    logger.info(f"Loading schemas from {tables_path}")
    with open(tables_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    db_schemas = {}

    for db in data:
        db_id = db["db_id"]

        table_names = db["table_names_original"]
        column_names = db["column_names_original"]
        foreign_keys = db["foreign_keys"]

        tables = defaultdict(list)

        # Build table → columns
        for table_idx, col_name in column_names:
            if table_idx == -1:
                continue
            tables[table_names[table_idx]].append(col_name)

        # Build FK relations
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
            "graph": dict(graph),
        }

    return db_schemas

def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))


def find_relevant_tables(question, schema, db_id=""):
    """
    Match tables if:
    - table name in question
    - any column name in question
    """
    tokens = tokenize(question)
    matched = set()

    for table, cols in schema["tables"].items():
        if table.lower() in tokens:
            matched.add(table)
            continue

        for col in cols:
            if col.lower() in tokens:
                matched.add(table)
                break

    if not matched:
        logger.debug(f"[{db_id}] No tables matched for: '{question}'. Falling back to all tables.")
        return set(schema["tables"].keys())

    logger.debug(f"[{db_id}] Initial match: {matched}")
    return matched

def expand_with_foreign_keys(selected_tables, schema, max_hops=1, db_id=""):
    """
    Adds neighbor tables via FK graph to preserve join paths.
    """
    graph = schema["graph"]
    expanded = set(selected_tables)

    for i in range(max_hops):
        new_tables = set()
        for table in expanded:
            neighbors = graph.get(table, [])
            new_tables.update(neighbors)
        
        diff = new_tables - expanded
        if diff:
            logger.debug(f"[{db_id}] Hop {i+1} expanded: {diff}")
            expanded.update(new_tables)
        else:
            break

    return expanded

def format_schema(db_id, schema, selected_tables):
    lines = []
    lines.append(f"Database: {db_id}")
    lines.append("Tables:")

    for table in sorted(selected_tables):
        cols = schema["tables"][table]
        cols_str = ", ".join(cols)
        lines.append(f"{table}({cols_str})")

    # Foreign keys
    fk_lines = []
    for t1, c1, t2, c2 in schema["foreign_keys"]:
        if t1 in selected_tables and t2 in selected_tables:
            fk_lines.append(f"{t1}.{c1} -> {t2}.{c2}")

    if fk_lines:
        lines.append("Foreign Keys:")
        lines.extend(fk_lines)

    return "\n".join(lines)

def build_sft_prompt(db_id, schema, question):
    """
    Centralized prompt builder to ensure consistency between
    Data Prep, Training, and Evaluation.
    """
    # 1. Match tables
    matched = find_relevant_tables(question, schema, db_id)
    
    # 2. Expand for joins
    selected_tables = expand_with_foreign_keys(matched, schema, db_id=db_id)
    
    # 3. Format text
    schema_text = format_schema(db_id, schema, selected_tables)
    
    return f"{schema_text}\n\nQuestion: {question}\nSQL:"
