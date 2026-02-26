# Module Imports
from config import ScrapeConfig
## Rich Imports
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
## Pkgs Imports
import markdown
from bs4 import BeautifulSoup
from queue import deque
import tempfile
from urllib.parse import urlparse
from dataclasses import asdict
import logging
import requests
import re
import os
from exceptions import (
    PipelineError,
    ConfigurationError,
    ScrapeError,
 
    LLMResponseError,
)

logger = logging.getLogger(__name__)

class ScrapeUrl:
    def __init__(self, cfg: ScrapeConfig):
        self.cfg = cfg
        self.url = cfg.url
        self.isFlash = cfg.scrape_type == "flash"
        self.remove_temp_files = cfg.remove_temp_files 
        self.raw_file_path = None
        self.cleaned_file_path = None
        self.base_url = urlparse(self.url).netloc

    def display_config(self):
        console = Console()

        ## Creating table for the key-value pair
        table = Table(show_header=False, box=None, padding=(0,2))
        table.add_column("Key", style="bold cyan")
        table.add_column("Value", style="magenta")

        for k, v in asdict(self.cfg).items():
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

    def scrape_url(self):
        logging.info(f"Deciding Which scrape to run: Flash or Deep")
        if self.isFlash:
            logging.info(f"Running Flash Scrape...")
            self.raw_file_path = self.flash_scrape()
            return {"Flash": True, "Deep": False}
        logging.info("Running Deep Scrape...")
        self.raw_file_path = self.deep_scrape()
        return {"Flash": False, "Deep": True}

    def flash_scrape(self):
        logger.info(f"Scraping the {self.url}...")

        url = f"https://r.jina.ai/{self.cfg.url}"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
        }
        try:
            response = None
            response = requests.get(url, headers=headers, timeout=10)
        except Exception as e:
            raise RuntimeError(f"Unable to fetch the raw data from from {url}, with error {e}")
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(prefix="scraped_", suffix=".md", mode = 'w', delete=False) as f:
            f.write(response.text)
            self.raw_file_path = f.name

        logger.info(f"Scraped the {self.url} successfully!\nTemp File path: {self.raw_file_path}")
        return self.raw_file_path

    def deep_scrape(self):
        visited = set()
        queue = deque([self.url]) 

        with tempfile.NamedTemporaryFile(prefix="scraped_", suffix=".md", mode = 'a', delete=False) as f:
            while queue:
                current_url = queue.popleft()
                if current_url in visited:
                    logging.info(f"Skipping the {current_url}...")
                    continue
                visited.add(current_url)
                ## Config
                url = f"https://r.jina.ai/{current_url}"
                headers = {
                    "Authorization": f"Bearer {self.cfg.api_key}",
                }
                try:
                    response = None
                    try:
                        response = requests.get(url, headers=headers, timeout=10)
                    except Exception as e:
                        raise RuntimeError(f"Unable to fetch the raw data from from {url}, with error {e}")

                    response.raise_for_status() # This triggers an error for 404s or 500s
                except Exception as e:
                    raise ScrapeError(e, current_url, response.status_code)

                f.write(f"Url: {current_url}\n")
                f.write(response.text + "\n\n")

                links = self.extract_links_from_text(response.text)
                for link in links:
                    if link not in visited:
                        queue.append(link)

            self.raw_file_path = f.name
        logging.info(f"Scraped the {self.url} successfully!\nTemp File path: {self.raw_file_path}")
        return self.raw_file_path

    def clean_markdown(self):
        logging.info(f"Initializing the cleaning process on {self.raw_file_path}")
        with open(self.raw_file_path, 'r') as f:
            raw_text = f.read()

        ## Cleaning Markdown Images and svg
        cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', raw_text)

        # ## Removing standard markdown lines
        # cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)

        # ## Remove metadata headers (like "Url:", "Title:", "Published Time:")
        # cleaned = re.sub(r'^(Url|Title|Published Time):.*$', '', cleaned, flags=re.MULTILINE)

        with tempfile.NamedTemporaryFile(prefix="cleaned_", suffix=".md", mode = 'w', delete=False) as f:
            f.write(cleaned)
            self.cleaned_file_path = f.name

        logging.info(f"Cleaned the {self.raw_file_path} successfully! Cleaned File path: {self.cleaned_file_path}")
        return cleaned

    def extract_links_from_text(self, markdown_text):
        logging.info("Initializing the Links extraction process...")

        html = markdown.markdown(markdown_text)
        soup = BeautifulSoup(html, "html.parser")

        links = set()
        for a in soup.find_all("a", href=True):
            link = a["href"]
            # Check if the link belongs to the same domain and isn't just an anchor (#)
            if urlparse(link).netloc == self.base_url and "#" not in link:
                links.add(link)

        logger.info(f"Extracted {len(links)} internal links.")
        return links

    def cleanup(self):
        """Deletes temporary files created during the process."""
        for path in [self.raw_file_path, self.cleaned_file_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Deleted temporary file: {path}")
                except Exception as e:
                    logger.error(f"Failed to delete {path}: {e}")
