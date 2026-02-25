import logging
import os
import markdown
from bs4 import BeautifulSoup
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
import tempfile
from urllib.parse import urlparse
# Load environment variables
load_dotenv()

logger = logging.getLogger("__main__")

def chunk_hash(chunk: str, algorithm: str = "sha256"):
    try:
        hash_obj = hashlib.new(algorithm)
    except Exception as e:
        raise ProcessingError(f"Unable to find the hash algorithm {algorithm}, with error {e}")
    
    chunk_bytes = chunk.encode('utf-8')
    hash_obj.update(chunk_bytes)
    
    return hash_obj.hexdigest()

class ScrapeUrl:
    def __init__(self, cfg: ScrapeConfig):
        self.cfg =cfg
        self.url = cfg.url
        self.isFlash = cfg.scrape_type == "flash"
        self.raw_file_path = None
        self.cleaned_file_path = None
        self.base_url = urlparse(self.url).netloc

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

        with tempfile.NamedTemporaryFile(prefix="scrape_", suffix=".md", mode = 'w', delete=False) as f:
            f.write(response.text)
            self.raw_file_path = f.name

        logger.info(f"Scraped the {self.url} successfully!\nTemp File path: {self.raw_file_path}")
        return self.raw_file_path

    def deep_scrape(self):
        if self.cfg.scrape_type == "flash":
            self.raw_file_path = self.flash_scrape()
            return self.raw_file_path

        visited = set()
        queue = deque([self.cfg.url]) 

        with tempfile.NamedTemporaryFile(prefix="scrape_", suffix=".md", mode = 'a', delete=False) as f:
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
                    try:
                        response = None
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

if __name__ == "__main__":
    from main import load_config
    my_config = load_config()
    scrapper = ScrapeUrl(my_config['web_scrapping'])
    try: 
        raw_path = scrapper.deep_scrape()
        cleaned_path = scrapper.clean_markdown()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    # finally:
        # scrapper.cleanup()