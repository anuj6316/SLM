"""
SQL utilities for cleaning and processing SQL strings.
"""


def clean_sql(sql: str) -> str:
    """
    Clean generated SQL output.

    Args:
        sql: Raw SQL string

    Returns:
        Cleaned SQL string
    """
    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql = " ".join(sql.split())

    end_markers = ["<|im_end|>", "###", "\n\n"]
    for marker in end_markers:
        if marker in sql:
            sql = sql.split(marker)[0].strip()

    return sql


def extract_question_from_input(input_text: str) -> str:
    """
    Extract question from formatted input.

    Args:
        input_text: Formatted input string with schema and question

    Returns:
        Extracted question string
    """
    if "### Question:" in input_text:
        return input_text.split("### Question:")[-1].strip().split("###")[0].strip()
    return input_text
