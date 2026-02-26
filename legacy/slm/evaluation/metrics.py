"""
Evaluation Metrics - Accuracy calculation functions.
"""

from typing import List, Tuple


def calculate_exact_match(
    predictions: List[str],
    gold_queries: List[str],
) -> Tuple[int, float]:
    """
    Calculate exact match accuracy.

    Args:
        predictions: List of predicted SQL queries
        gold_queries: List of gold SQL queries

    Returns:
        Tuple of (exact_matches, accuracy)
    """
    exact_matches = sum(
        1
        for p, g in zip(predictions, gold_queries)
        if p.lower().strip() == g.lower().strip()
    )
    accuracy = exact_matches / len(predictions) if predictions else 0.0
    return exact_matches, accuracy


def extract_question(entry: dict) -> str:
    """
    Extract question from a dataset entry.

    Args:
        entry: Dataset entry dict

    Returns:
        Question string
    """
    input_text = entry.get("input", "")
    if "### Question:" in input_text:
        return input_text.split("### Question:")[-1].strip().split("###")[0].strip()
    return entry.get("question", "")
