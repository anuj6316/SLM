from utils import load_config
from scrape_flow import ScrapeUrl
from pathlib import Path
from config import ScrapeConfig, ProcessMarkdownQAPairsConfig
from rich.logging import RichHandler
from utils import display_config, text_splitter
import logging 

logging.basicConfig(level="INFO", handlers=[RichHandler()])
logger = logging.getLogger("rich")

BASE_DIR = Path(__file__).parent
logging.info(f"Running this at {BASE_DIR} root path")

def run_scrapping(cfg: str = BASE_DIR/"config.yml"):
    scrape_cfg = cfg['web_scrapping']
    display_config(scrape_cfg)

    scrapper = ScrapeUrl(scrape_cfg)
    scrapper.scrape_url()
    scrapper.clean_markdown()
    # scrapper.cleanup()
    return scrapper.cleaned_file_path

def run_qa_generation(cfg: str = BASE_DIR/"config.yml"):
    cfg = load_config(cfg_path)
    qa_config = ProcessMarkdownQAPairsConfig(**cfg['ProcessMarkdownQAPairs'])
    display_config(qa_config)

    ## generate chunks
    chunks = text_splitter(data)

if __name__ == "__main__":
    run_scrapping()