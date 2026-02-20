"""
SQL Cleaner Module - Validates and normalizes SQL queries for SFT training.

Responsibilities:
  1. Syntactic Validation: Uses sqlglot to parse SQL, dropping malformed queries.
  2. Canonicalization: Converts all SQL to consistent single-line format.
  3. Token Efficiency: Removes extra whitespace/newlines.
  4. Sanitization: Cleans natural language questions.
"""
import logging
from typing import Optional, Dict, Any
import sqlglot

logger = logging.getLogger("cleaner")

_cleaner_instance: Optional["SQLCleaner"] = None


class SQLCleaner:
    """Validates and normalizes SQL queries for LLM training."""

    def __init__(self, dialect: str = "sqlite"):
        self.dialect = dialect

    def normalize_sql(self, sql_query: str) -> Optional[str]:
        """
        Validates SQL syntax and formats into canonical single-line form.

        Args:
            sql_query: Raw SQL query string

        Returns:
            Normalized SQL string, or None if invalid
        """
        if not sql_query or not isinstance(sql_query, str):
            logger.warning("Empty or non-string SQL encountered.")
            return None
        try:
            parsed = sqlglot.parse_one(sql_query, read=self.dialect)
            return parsed.sql(dialect=self.dialect, pretty=False)
        except Exception as e:
            logger.error(f"SQL Syntax Error: {e} | Query snippet: {sql_query[:100]}...")
            return None

    def sanitize_text(self, text: str) -> str:
        """
        Cleans natural language text by normalizing whitespace.

        Args:
            text: Raw text string

        Returns:
            Sanitized text with normalized whitespace
        """
        if not text:
            return ""
        return " ".join(text.split())

    def process_record(self, question: str, sql_query: str) -> Optional[Dict[str, str]]:
        """
        Main entry point for cleaning a single record.

        Args:
            question: Natural language question
            sql_query: SQL query string

        Returns:
            Dict with 'question' and 'sql' keys, or None if invalid
        """
        cleaned_sql = self.normalize_sql(sql_query)
        if not cleaned_sql:
            return None
        cleaned_question = self.sanitize_text(question)
        if not cleaned_question:
            logger.warning("Record dropped due to empty question")
            return None

        return {
            "question": cleaned_question,
            "sql": cleaned_sql
        }


def get_cleaner() -> SQLCleaner:
    """Returns singleton SQLCleaner instance."""
    global _cleaner_instance
    if _cleaner_instance is None:
        _cleaner_instance = SQLCleaner(dialect="sqlite")
    return _cleaner_instance


def clean_and_validate(raw_entry: Dict[str, Any], dataset_type: str) -> Optional[Dict[str, Any]]:
    """
    Adapter function to clean raw dataset entries.

    Args:
        raw_entry: Raw dataset record
        dataset_type: One of 'spider', 'bird', 'gretel'

    Returns:
        Entry with 'cleaned_sql' and sanitized 'question', or None if invalid
    """
    cleaner = get_cleaner()

    sql_key_map: Dict[str, str] = {
        "spider": "query",
        "bird": "SQL",
        "gretel": "sql"
    }
    question_key_map: Dict[str, str] = {
        "spider": "question",
        "bird": "question",
        "gretel": "sql_prompt"
    }

    sql_key = sql_key_map.get(dataset_type, "query")
    question_key = question_key_map.get(dataset_type, "question")

    if sql_key not in raw_entry and "sql" in raw_entry:
        sql_key = "sql"

    raw_sql = raw_entry.get(sql_key, "")
    raw_question = raw_entry.get(question_key, "")

    result = cleaner.process_record(raw_question, raw_sql)

    if result:
        raw_entry['cleaned_sql'] = result['sql']
        raw_entry['question'] = result['question']
        return raw_entry

    return None


if __name__ == "__main__":
    test_entry = {
        "query": "SELECT  *  FROM  users  WHERE  id = 1",
        "question": "  Find all users   with id 1  "
    }
    result = clean_and_validate(test_entry, "spider")
    if result:
        print(f"Cleaned SQL: {result['cleaned_sql']}")
        print(f"Cleaned Question: {result['question']}")
    else:
        print("Validation failed")
