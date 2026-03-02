from utils import load_config
from scrape_flow import ScrapeUrl
from pathlib import Path
from config import ScrapeConfig, ProcessMarkdownQAPairsConfig
from rich.logging import RichHandler
from utils import display_config, text_splitter
from qa_generator import ProcessMarkdownQAPairs
from tqdm.auto import tqdm
import os
import logging 

logging.basicConfig(level="INFO", handlers=[RichHandler()])
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_core").setLevel(logging.WARNING)
logging.getLogger("langchain_text_splitter").setLevel(logging.WARNING)
logging.getLogger("langchain_groq").setLevel(logging.WARNING)
logger = logging.getLogger("rich")


BASE_DIR = Path(__file__).parent
logging.info(f"Running this at {BASE_DIR} root path")

def run_scrapping(cfg: ScrapeConfig):
    display_config(cfg)

    scrapper = ScrapeUrl(cfg)
    scrapper.scrape_url()
    scrapper.clean_markdown()
    # scrapper.cleanup() # Don't cleanup yet
    return scrapper.cleaned_file_path

def run_qa_generation(cfg: ProcessMarkdownQAPairsConfig, file_path: str):
    # Update the file path to the one we just scraped
    cfg.file_path = file_path
    display_config(cfg)
    
    qa_obj = ProcessMarkdownQAPairs(cfg)
    data = qa_obj.read_file(file_path)
    ## generate chunks
    chunks = text_splitter(data)
    # for chunk in tqdm(range(len(chunks)), desc="Processing Chunks for generating QA Pairs: "):
    #     try
    qa_obj.run()

if __name__ == "__main__":
    ## global config
    cfg = load_config(BASE_DIR/"config.yml")
    
    ## Scrapping
    scrape_cfg = ScrapeConfig(**cfg['web_scrapping'])
    cleaned_path = run_scrapping(scrape_cfg)
    
    ## Dataset generator
    qa_cfg = ProcessMarkdownQAPairsConfig(**cfg['ProcessMarkdownQAPairs'])
    run_qa_generation(qa_cfg, cleaned_path)