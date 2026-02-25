from utils import get_raw_content, bfs_crawl, cleaning_raw_markdown_content
import yaml
from pprint import pprint
import os
from dotenv import load_dotenv

# config
from config import ScrapeConfig
load_dotenv()

def load_config(path: str = "/home/mindmap/Desktop/SLM/unstructured_data/config.yml"):
    with open(path, "r") as f:
        content = os.path.expandvars(f.read())  # <-- expands ${VAR}
        cfg = yaml.safe_load(content)
    return cfg

def web_scrapping_logic(cfg):
    return f"Scraping {cfg.url} with key {cfg.api_key[:4]}..."

def run_scraper(config_path="/home/mindmap/Desktop/SLM/unstructured_data/config.yml"):
    cfg = load_config(config_path)
    cfg = scrapeConfig(**cfg['web_scrapping'])
    result = web_scrapping_logic(cfg)
    return result

if __name__ == "__main__":
    # if not os.path.exists("flash_")
    cfg = load_config()
    pprint(cfg)
    scrape_config = ScrapeConfig(**cfg['web_scrapping'])
    bfs_crawl(scrape_config)
    with open("flash_scrape.md", 'r') as f:
        raw_text = f.read()
    cleaning_raw_markdown_content(raw_text)
    # cleaning_raw_markdown_content(scrape_config)


    # raw_text = get_raw_content(scrape_config)
    # pprint(raw_text.links)
    # links = extract_links_from_markdown(raw_text.content_path)
