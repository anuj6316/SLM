import yaml
import argparse
import logging
import os
import sys
from typing import Generator, Dict, Any

# Add src to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from preprocessing import (
    create_loader,
    SFTProcessor,
    JsonlFormatter
)
from core.utils import get_logger, console
from core.config import load_config, AppConfig
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.live import Live

console = Console()

logger = get_logger("Preprocessor")

def run_pipeline(config: AppConfig):
    """
    Executes the Load -> Process -> Format pipeline using the provided AppConfig.
    """
    console.print(Panel("[bold blue]SLM Preprocessing Pipeline[/]", expand=False))
    
    # Initialize Components
    try:
        data_config = config.data
        formatting_config = config.get('formatting', {})
        paths_config = config.paths

        # Convert AppConfig back to dict for the components that expect dicts
        data_dict = vars(data_config) if isinstance(data_config, AppConfig) else data_config
        formatting_dict = vars(formatting_config) if isinstance(formatting_config, AppConfig) else formatting_config

        loader = create_loader(data_dict)
        processor = SFTProcessor(data_dict)
        formatter = JsonlFormatter(formatting_dict)
        output_path = paths_config.data_path
    except (AttributeError, KeyError) as e:
        console.print(f"[bold red]Initialization Failed:[/] Invalid or missing configuration: {str(e)}")
        logger.error(f"Initialization Failed: Invalid or missing configuration: {str(e)}")
        raise

    # Execute Pipeline with Progress
    total_rows = loader.get_row_count()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        load_task = progress.add_task("[cyan]Reading source...", total=total_rows)
        proc_task = progress.add_task("[magenta]Processing rows...", total=total_rows)
        write_task = progress.add_task("[yellow]Writing to disk...", total=total_rows)

        def load_wrapper():
            for item in loader.load():
                progress.advance(load_task)
                yield item
        
        def proc_wrapper(stream):
            for item in processor.process(stream):
                progress.advance(proc_task)
                yield item

        def write_wrapper(stream):
            for item in stream:
                progress.advance(write_task)
                yield item

        raw_stream = load_wrapper()
        processed_stream = proc_wrapper(raw_stream)
        tracked_stream = write_wrapper(processed_stream)
        
        formatter.format(tracked_stream, output_path)

    # Summary Table
    summary = Table(title="Pipeline Execution Summary", show_header=True, header_style="bold magenta")
    summary.add_column("Metric", style="dim")
    summary.add_column("Value", justify="right")
    
    summary.add_row("Input Source", data_dict.get('path', 'Unknown'))
    summary.add_row("Output Path", output_path)
    summary.add_row("Source Rows", str(total_rows))
    
    console.print(summary)
    console.print("\n[bold green]✅ Pipeline completed successfully![/]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLM Data Preprocessing CLI")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    # Load the structured config
    app_config = load_config(args.config)
    run_pipeline(app_config)