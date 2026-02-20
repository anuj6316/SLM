"""
Shared utilities for SLM Text-to-SQL pipeline.

Public API:
    setup_logging: Configure logging with Rich handler
    get_logger: Get a logger for a module
    load_schema_dict: Load database schemas from tables.json
    clean_sql: Clean generated SQL output
    extract_question_from_input: Extract question from formatted input
"""

from slm.utils.logging import setup_logging, get_logger
from slm.utils.schema import load_schema_dict
from slm.utils.sql import clean_sql, extract_question_from_input

__all__ = [
    "setup_logging",
    "get_logger",
    "load_schema_dict",
    "clean_sql",
    "extract_question_from_input",
]
