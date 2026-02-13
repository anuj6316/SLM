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

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.live import Live

console = Console()

# Setup Logging - Keep only errors/criticals in console to avoid interfering with Rich
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Preprocessor")

# Filter out lower level logs from console stream to keep UI clean
class RichFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING

def run_pipeline(config_path: str):
    """
    Executes the Load -> Process -> Format pipeline with Rich UI.
    """
    console.print(Panel("[bold blue]SLM Preprocessing Pipeline[/]", expand=False))
    
    # 1. Load Config
    with console.status("[bold green]Loading configuration...") as status:
        if not os.path.exists(config_path):
            console.print(f"[bold red]Error:[/] Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # 2. Initialize Components
    try:
        loader = create_loader(config['data'])
        processor = SFTProcessor(config['data'])
        formatter = JsonlFormatter(config.get('formatting', {}))
        output_path = config['paths']['data_path']
    except Exception as e:
        console.print(f"[bold red]Initialization Failed:[/] {str(e)}")
        sys.exit(1)

    # 3. Execute Pipeline with Progress
    total_rows = loader.get_row_count()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        # Define Tasks
        load_task = progress.add_task("[cyan]Reading source...", total=total_rows)
        proc_task = progress.add_task("[magenta]Processing rows...", total=total_rows)
        write_task = progress.add_task("[yellow]Writing to disk...", total=total_rows)

        # We wrap the streams manually to update the progress bars
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

        # Execute
        raw_stream = load_wrapper()
        processed_stream = proc_wrapper(raw_stream)
        tracked_stream = write_wrapper(processed_stream)
        
        formatter.format(tracked_stream, output_path)

    # 4. Summary Table
    summary = Table(title="Pipeline Execution Summary", show_header=True, header_style="bold magenta")
    summary.add_column("Metric", style="dim")
    summary.add_column("Value", justify="right")
    
    summary.add_row("Input Source", config['data']['path'])
    summary.add_row("Output Path", output_path)
    summary.add_row("Source Rows", str(total_rows))
    # Note: formatter logs the actual count, we can capture it if we want but for now showing the flow
    
    console.print(summary)
    console.print("\n[bold green]✅ Pipeline completed successfully![/]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLM Data Preprocessing CLI")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)
