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
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, # Set log level to DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s", # Define log format
    datefmt="%Y-%m-%d %H:%M:%S" # Customize timestamp format
)
logger = logging.getLogger("__main__")

def chunk_hash(chunk: str, algorithm: str = "sha256"):
    try:
        hash_obj = hashlib.new(algorithm)
    except Exception as e:
        raise ProcessingError(f"Unable to find the hash algorithm {algorithm}, with error {e}")
    
    chunk_bytes = chunk.encode('utf-8')
    hash_obj.update(chunk_bytes)
    
    return hash_obj.hexdigest()

def get_raw_content(cfg: ScrapeConfig):
    url = f"https://r.jina.ai/{cfg.url}"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
    }
    try:
        response = None
        response = requests.get(url, headers=headers, timeout=10)
    except Exception as e:
        raise RuntimeError(f"Unable to fetch the raw data from from {url}, with error {e}")
    response.raise_for_status()

    with tempfile.NamedTemporaryFile("flash_scrape.md", 'w', delete=False) as f:
        f.write(response.text)
        

    hsh = chunk_hash(response.text)[:3]
    return f"./flash_scrape_{hsh}.md"

def extract_links_from_text(markdown_text):
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")

    links = {
        a["href"]
        for a in soup.find_all("a", href=True)
        if a["href"].startswith("https://www.mindmapdigital.ai")
        and "#" not in a["href"]
    }

    return links

def bfs_crawl(cfg: ScrapeConfig):
    if cfg.scrape_type == "flash":
        raw_text, content_path = get_raw_content(cfg)
        return raw_text, content_path

    visited = set()
    queue = deque([cfg.url]) 

    while queue:
        current_url = queue.popleft()
        if current_url in visited:
            logging.info(f"Skipping the {current_url}...")
            continue
        visited.add(current_url)
        ## Config
        url = f"https://r.jina.ai/{current_url}"
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
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

        with open("bfs_output.md", 'a') as f:
            f.write(f"Url: {current_url}\n")
            f.write(response.text + "\n\n")

        links = extract_links_from_text(response.text)
        for link in links:
            if link not in visited:
                queue.append(link)

    return response.text, "bfs_output.md"

def cleaning_raw_markdown_content(raw_text: str):
    ## Cleaning Markdown Images and svg
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', raw_text)

    # ## Removing standard markdown lines
    # cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)

    # ## Remove metadata headers (like "Url:", "Title:", "Published Time:")
    # cleaned = re.sub(r'^(Url|Title|Published Time):.*$', '', cleaned, flags=re.MULTILINE)

    with open("cleaned_output.md", 'w') as f:
        f.write(cleaned)

    return cleaned

if __name__ == "__main__":
    with open("bfs_output.md", 'r') as f:
        raw_text = f.read()
    cleaning_raw_markdown_content(raw_text)
