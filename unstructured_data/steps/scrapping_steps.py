from zenml import step
from utils import load_config
from  pathlib import Path
from config import ScrapeConfig

@step
def load_config_step(cfg_path: str):
    return load_config(cfg_path)

@step
def scrape_url_step(cfg: dict):
    scrape_cfg = ScrapeConfig(**cfg['web_scrapping'])
    scrapper = ScrapeUrl(scrape_cfg)
    scrapper.scrape_url()
    return scrape_cfg

@step
def cleanup_step(scrape_cfg: ScrapeConfig):
    scrapper = ScrapeUrl(scrape_cfg)
    scrapper.cleanup()
    # return scrape_cfg