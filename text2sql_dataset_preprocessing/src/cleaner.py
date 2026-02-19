"""
Purpose: This is a "Service Class" that performs the actual heavy lifting of cleaning. It doesn't care where the data comes from (Spider, BIRD, or a database); it only knows how to clean SQL and Text.

What the Cleaner Achieves
 1. Syntactic Validation (The Filter): It uses sqlglot to parse the SQL. If a dataset has a typo or broken SQL, the cleaner catches it and drops the record. This ensures
    your model never learns "bad" SQL.
 2. Canonicalization (The Uniform): In raw datasets, one person might write SELECT * FROM table and another writes select * from table. The cleaner converts all of them into
    a single, consistent style. This reduces "noise" so the model learns the logic of SQL rather than just memorizing strings.
 3. Token Efficiency: By "one-lining" the SQL and removing extra whitespace/newlines, you use fewer tokens. This allows you to fit more examples into a single training batch
    and speeds up training.
 4. Sanitization: It ensures the natural language question doesn't have hidden characters or messy spacing that could confuse the model's attention mechanism.
"""
import logging
import sqlglot

logger = logging.getLogger("cleaner")

class SQLCleaner:
    def __init__(self, dialect="sqlite"):
        self.dialect = dialect #A SQL dialect is a variation of SQL used by different databases.
    
    def normalize_sql(self, sql_query: str):
        """Validates that the SQL is syntactically correct and formats it into a standard "canonical" form."""
        if not sql_query or not isinstance(sql_query, str):
            logger.warning(f"Empty or non-string SQL encountered.")
            return None
        try:
            # sqlglot.parse_one ensures the sql is syntactically correct for the dialect
            parsed = sqlglot.parse_one(sql_query, read=self.dialect)
            # .sql(pretty=False) ensures a consistent, single-line format for llm training
            return parsed.sql(dialect=self.dialect, pretty=False)
        except Exception as e:
            logger.error(f"SQL Syntax Error: {e} | Query snippet: {sql_query[:100]}...")
            return None

    def sanitize_text(self, text: str):
        """Cleans up the "Natural Language" part of the data (the question)."""
        if not text:
            return ""
        return " ".join(text.split())

    def process_record(self, question: str, sql_query: str):
        """Main entry point for cleaning a single record"""
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

def clean_and_validate(raw_entry, dataset_type: str):
    """Adapter function to integrate with existing pipeline while using the new SQLCleaner"""
    cleaner = SQLCleaner(dialect="sqlite")
    sql_key_map = {
        "spider": "query",
        "bird": "SQL", 
        "gretel": "sql"
    }
    question_key = {
        "spider": "question",
        "bird": "question",
        "gretel": "sql_prompt"
    }

    # fallback logic if keys change in future datasets
    sql_key = sql_key_map.get(dataset_type, "query")
    question_key = question_key.get(dataset_type, "question")
    if sql_key not in raw_entry and "sql" in raw_entry:
        sql_key = "sql"
    
    raw_sql = raw_entry.get(sql_key, "")
    raw_question = raw_entry.get(question_key, "")
    
    result = cleaner.process_record(raw_question, raw_sql)
    
    if result:
        # Update entry with cleaned versions
        raw_entry['cleaned_sql'] = result['sql']
        raw_entry['question'] = result['question']
        return raw_entry
    
    return None

if __name__ == "__main__":
    # cleaner = SQLCleaner("sqlite")
    clean_and_validate()