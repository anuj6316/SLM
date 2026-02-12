import logging
from rich.logging import RichHandler
from rich.console import Console

# Shared console for rich output
console = Console()

def get_logger(name="sft-pipeline"):
    """Configures a logger with RichHandler for beautiful CLI output."""
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console, show_path=False)]
    )
    return logging.getLogger(name)
