"""
SQL utilities for cleaning and processing SQL strings.
"""


import re

def clean_sql(sql: str) -> str:
    """
    Clean generated SQL output by extracting code blocks or cleaning raw text.
    """
    # 1. Try to extract from ```sql or ``` blocks
    code_block_match = re.search(r"```(?:sql)?\n?(.*?)\n?```", sql, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        sql = code_block_match.group(1).strip()
    
    # 2. Basic cleanup
    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql = " ".join(sql.split())

    # 3. Handle truncated outputs or model chat prefixes
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
