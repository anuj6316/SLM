import re
from typing import Dict, Any, List

def build_prompt(
    config: "AppConfig",
    instruction: str,
    system_prompt: str = None
) -> str:
    """
    Builds a prompt for the model.

    Args:
        config: The application configuration.
        instruction: The instruction for the model.
        system_prompt: The system prompt to use.

    Returns:
        The formatted prompt.
    """
    if system_prompt is None:
        system_prompt = config.formatting.system_prompt

    return (
        f"<|im_start|>system
{system_prompt}<|im_end|>
"
        f"<|im_start|>user
{instruction}<|im_end|>
"
        f"<|im_start|>assistant
"
    )

def clean_sql(sql: str) -> str:
    """
    Cleans the generated SQL.

    Args:
        sql: The SQL to clean.

    Returns:
        The cleaned SQL.
    """
    sql = sql.split("assistant
")[-1].strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql = " ".join(sql.split())
    return sql

def format_example(example: Dict[str, Any], tokenizer: Any, config: "AppConfig") -> Dict[str, str]:
    """
    Formats an example for training.
    """
    if "messages" in example:
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}
    elif "instruction" in example and "output" in example:
        messages = [
            {"role": "system", "content": config.formatting.system_prompt},
            {"role": "user", "content": example.get("instruction", "")},
            {"role": "assistant", "content": example.get("output", "")}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
    return {"text": ""}
