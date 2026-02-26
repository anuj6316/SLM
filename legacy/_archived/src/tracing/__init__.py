"""
MLflow Tracing Module for Text-to-SQL

Provides traced inference with observability via Databricks MLflow.
"""

from .inference_databricks import (
    generate_sql,
    load_schema,
    build_prompt,
    text_to_sql_pipeline,
)

__all__ = [
    "generate_sql",
    "load_schema",
    "build_prompt",
    "text_to_sql_pipeline",
]
