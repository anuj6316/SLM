# Module Imports
from config import ScrapeConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

## Pkgs Imports
import threading
import markdown
from bs4 import BeautifulSoup
from queue import deque
import tempfile
from urllib.parse import urlparse
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
        self.api_key = cfg.api_key
        self.max_depth = cfg.max_depth
        self.max_workers = cfg.max_workers
        self.remove_temp_files = cfg.remove_temp_files 
        self.raw_file_path = None
        self.cleaned_file_path = None
        self.base_url = urlparse(self.url).netloc

    def scrape_url(self):
        logging.info(f"Deciding Which scrape to run: Flash or Deep")
        if self.isFlash:
            logging.info(f"Running Flash Scrape...")
            self.raw_file_path = self.flash_scrape()
            return {"Flash": True, "Deep": False}
        logging.info("Running Deep Scrape...")
        self.raw_file_path = self.deep_scrape(self.max_depth, self.max_workers)
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

    # def deep_scrape(self, max_depth: int = 2, max_workers: int = 5):
    #     visited = set()
    #     visited_lock = threading.Lock()

    #     ## (url, depth)
    #     queue = deque([self.url, 0]) 

    #     file_lock = threading.Lock()

    #     with tempfile.NamedTemporaryFile(prefix="scraped_", suffix=".md", mode = 'a', delete=False) as f:
    #         while queue:
    #             current_url = queue.popleft()
    #             if current_url in visited:
    #                 logging.info(f"Skipping the {current_url}...")
    #                 continue
    #             visited.add(current_url)
    #             ## Config
    #             url = f"https://r.jina.ai/{current_url}"
    #             headers = {
    #                 "Authorization": f"Bearer {self.cfg.api_key}",
    #             }
    #             try:
    #                 response = None
    #                 try:
    #                     response = requests.get(url, headers=headers, timeout=10)
    #                 except Exception as e:
    #                     raise RuntimeError(f"Unable to fetch the raw data from from {url}, with error {e}")

    #                 response.raise_for_status() # This triggers an error for 404s or 500s
    #             except Exception as e:
    #                 raise ScrapeError(e, current_url, response.status_code)

    #             f.write(f"Url: {current_url}\n")
    #             f.write(response.text + "\n\n")

    #             links = self.extract_links_from_text(response.text)
    #             for link in links:
    #                 if link not in visited:
    #                     queue.append(link)

    #         self.raw_file_path = f.name
    #     logging.info(f"Scraped the {self.url} successfully!\nTemp File path: {self.raw_file_path}")
    #     return self.raw_file_path

    def deep_scrape(self, max_depth=2, max_workers=5):
        visited = set()
        visited_lock = threading.Lock()

        # (url, depth)
        queue = deque([(self.url, 0)])

        file_lock = threading.Lock()

        with tempfile.NamedTemporaryFile(prefix="scraped_", suffix=".md", mode="a", delete=False) as f:

            def fetch_and_process(current_url, depth):
                nonlocal visited

                with visited_lock:
                    if current_url in visited:
                        logging.info(f"Skipping {current_url}")
                        return []
                    visited.add(current_url)

                if depth > max_depth:
                    return []

                url = f"https://r.jina.ai/{current_url}"
                headers = {
                    "Authorization": f"Bearer {self.cfg.api_key}",
                }

                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                except Exception as e:
                    logging.error(f"Error scraping {current_url}: {e}")
                    return []

                # Write safely
                with file_lock:
                    f.write(f"Url: {current_url}\n")
                    f.write(response.text + "\n\n")

                # Extract links for next level
                if depth < max_depth:
                    links = self.extract_links_from_text(response.text)
                    return [(link, depth + 1) for link in links]

                return []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                # initial task
                futures.append(executor.submit(fetch_and_process, self.url, 0))

                while futures:
                    for future in as_completed(futures):
                        futures.remove(future)
                        new_links = future.result()

                        for link, depth in new_links:
                            futures.append(
                                executor.submit(fetch_and_process, link, depth)
                            )

        self.raw_file_path = f.name
        logging.info(f"Scraped {self.url} successfully!\nTemp File path: {self.raw_file_path}")
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