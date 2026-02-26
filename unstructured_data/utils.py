import logging
import os
from dotenv import load_dotenv
from config import ScrapeConfig
from pprint import pprint
from collections import deque 
import requests
import re
import hashlib
from exceptions import (
    PipelineError,
    ConfigurationError,
    ScrapeError,
    LLMResponseError,
    ProcessingError,
    RateLimitError
)
## Rich Imports
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dataclasses import asdict

import yaml
# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger("__main__")

def chunk_hash(chunk: str, algorithm: str = "sha256"):
    try:
        hash_obj = hashlib.new(algorithm)
    except Exception as e:
        raise ProcessingError(f"Unable to find the hash algorithm {algorithm}, with error {e}")
    
    chunk_bytes = chunk.encode('utf-8')
    hash_obj.update(chunk_bytes)
    
    return hash_obj.hexdigest()

def load_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        content = f.read()
    content = os.path.expandvars(content)
    return yaml.safe_load(content)

def text_splitter(document):
    """Creating multiple chunks form the raw text"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(document)
    return texts

def display_config(cfg):
    console = Console()

    ## Creating table for the key-value pair
    table = Table(show_header=False, box=None, padding=(0,2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="magenta")

    for k, v in asdict(cfg).items():
        if v is not None:
            table.add_row(k.replace("_", " ").title(), str(v))

    # Wrap the table in a pretty Panel
    print(
        Panel(
            table,
            title="[bold green]Application Configuration[/bold green]",
            border_style="bright_blue",
            width=60,
            expand=False,
            padding=(1, 2)
        )
    )  
if __name__ == "__main__":
    from main import load_config
    my_config = load_config()
    pprint(my_config['web_scrapping'])
    scrapper = ScrapeUrl(my_config['web_scrapping'])
    # pprint(scrapper)
    try: 
        raw_path = scrapper.deep_scrape()
        cleaned_path = scrapper.clean_markdown()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    # finally:
        # scrapper.cleanup()