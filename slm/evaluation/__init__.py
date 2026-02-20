"""
Evaluation Module - Evaluate Text-to-SQL models on benchmarks.

Public API:
    Evaluator: Evaluate models on Spider and custom datasets
    calculate_exact_match: Calculate exact match accuracy
    extract_question: Extract question from dataset entry
"""

from slm.evaluation.evaluator import Evaluator
from slm.evaluation.metrics import calculate_exact_match, extract_question

__all__ = [
    "Evaluator",
    "calculate_exact_match",
    "extract_question",
]
