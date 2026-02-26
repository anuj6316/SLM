"""
Logging utilities for SLM Text-to-SQL pipeline.
"""

import logging
from typing import Optional

from rich.logging import RichHandler


def setup_logging(
    name: str = "slm",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup logging with Rich handler.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
