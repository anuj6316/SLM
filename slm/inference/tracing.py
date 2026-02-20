"""
MLflow Tracing - Utilities for tracing inference with MLflow/Databricks.
"""

import logging
from typing import Any, Callable, TypeVar, Optional

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def setup_mlflow(tracking_uri: str, experiment_name: str) -> bool:
    """
    Configure MLflow for tracing.

    Args:
        tracking_uri: MLflow tracking URI (e.g., "databricks")
        experiment_name: Name of the experiment

    Returns:
        True if setup successful, False otherwise
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracing enabled: {experiment_name}")
        return True
    except ImportError:
        logger.warning("MLflow not installed. Tracing disabled.")
        return False


def traced_generate(
    generate_fn: Callable[..., Any],
    question: str,
    db_id: str,
    mlflow_enabled: bool,
    prompt: Optional[str] = None,
) -> Any:
    """
    Generate SQL with optional MLflow tracing.

    Args:
        generate_fn: The generate function to call
        question: Natural language question
        db_id: Database identifier
        mlflow_enabled: Whether MLflow tracing is enabled
        prompt: Optional pre-formatted prompt

    Returns:
        Result from generate_fn
    """
    if not mlflow_enabled:
        return generate_fn(question, db_id, prompt=prompt)

    import mlflow

    @mlflow.trace(name="generate_sql", span_type="LLM")
    def _trace_generate():
        result = generate_fn(question, db_id, prompt=prompt)
        mlflow.log_metrics(
            {
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
            }
        )
        return result

    return _trace_generate()
