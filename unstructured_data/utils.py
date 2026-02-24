import logging
import os
import markdown
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from config import scrapeConfig
from pprint import pprint
from collections import deque 
import requests
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, # Set log level to DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s", # Define log format
    datefmt="%Y-%m-%d %H:%M:%S" # Customize timestamp format
)
logger = logging.getLogger("__main__")

def get_raw_content(cfg: scrapeConfig, visited=None):
    """

    """
    try:
        import requests
        
        # tracking visited url
        if visited is None:
            visited = set()
        
        # working url
        current_url = cfg.url if cfg.active_url is None else cfg.active_url

        # is current url already been scrapped
        if current_url in visited:
            logging.info(f"Skipping the {current_url}...")
            return
        visited.add(current_url)
        ## Config
        url = f"https://r.jina.ai/{current_url}"
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
        }

        response = requests.get(url, headers=headers)

        # Optionally, you can check the response status or print the response content
        # print(response.status_code)
        logger.info(f"{'='*50}\nResponse Code: {response.status_code} | Url: {current_url}\n{'='*50}\n")

        with open("output.md", 'a') as f:
            f.write(f"Url: {current_url}\n")
            f.write(response.text + "\n\n")

        # 🔥 Extract links from THIS PAGE ONLY
        links = extract_links_from_text(response.text)

        for link in links:
            if link not in visited:
                cfg.active_url = link
                get_raw_content(cfg, visited)
    except Exception as e:
        raise RuntimeError(f"Unable to fetch the raw data from from {url}, with error {e}")

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

def bfs_crawl(cfg: scrapeConfig):
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
        response = requests.get(url, headers=headers)

        with open("bfs_output.md", 'a') as f:
            f.write(f"Url: {current_url}\n")
            f.write(response.text + "\n\n")

        links = extract_links_from_text(response.text)
        for link in links:
            if link not in visited:
                queue.append(link)

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
