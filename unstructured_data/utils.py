import logging
import os
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

import yaml
# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger("__main__")

def chunk_hash(chunk: str, algorithm: str = "sha256"):
    try:
        hash_obj = hashlib.new(algorithm)
    except Exception as e:
        raise ProcessingError(f"Unable to find the hash algorithm {algorithm}, with error {e}")
    
    chunk_bytes = chunk.encode('utf-8')
    hash_obj.update(chunk_bytes)
    
    return hash_obj.hexdigest()

def load_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        content = f.read()
    content = os.path.expandvars(content)
    return yaml.safe_load(content)

if __name__ == "__main__":
    from main import load_config
    my_config = load_config()
    pprint(my_config['web_scrapping'])
    scrapper = ScrapeUrl(my_config['web_scrapping'])
    # pprint(scrapper)
    try: 
        raw_path = scrapper.deep_scrape()
        cleaned_path = scrapper.clean_markdown()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    # finally:
        # scrapper.cleanup()