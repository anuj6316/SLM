from utils import load_config
from scrape_flow import ScrapeUrl
from pathlib import Path
from config import ScrapeConfig
from rich.logging import RichHandler
import logging 

logging.basicConfig(level="INFO", handlers=[RichHandler()])
logger = logging.getLogger("rich")

BASE_DIR = Path(__file__).parent
logging.info(f"Running this at {BASE_DIR} root path")

def run_scrapping(cfg_path: str = BASE_DIR/"config.yml"):
    cfg = load_config(cfg_path)
    scrape_cfg = ScrapeConfig(**cfg['web_scrapping'])
    scrapper = ScrapeUrl(scrape_cfg)
    scrapper.display_config()
    scrapper.scrape_url()
    scrapper.clean_markdown()
    scrapper.cleanup()


if __name__ == "__main__":
    run_scrapping()