"""
Inference Module - Local and MLflow-traced inference for Text-to-SQL.

Public API:
    InferenceEngine: Main inference engine with optional MLflow tracing
    setup_mlflow: Configure MLflow for Databricks tracing
"""

from slm.inference.engine import InferenceEngine
from slm.inference.tracing import setup_mlflow

__all__ = [
    "InferenceEngine",
    "setup_mlflow",
]
