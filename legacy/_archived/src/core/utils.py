import logging
from rich.logging import RichHandler
from rich.console import Console

# Shared console for rich output
console = Console()

def get_logger(name="slm-pipeline"):
    """
    Configures and returns a logger with RichHandler for beautiful CLI output.
    
    Args:
        name (str): The name of the logger. Defaults to "slm-pipeline".
        
    Returns:
        A configured logger instance.
    """
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console, show_path=False)]
    )
    return logging.getLogger(name)
