import sys
import os

# Add the parent folder (project root) to sys.path
sys.path.append(os.path.abspath(".."))

from zenml import pipeline
from steps.scrapping_steps import (
    load_config_step,
    scrape_url_step as scrape_step,
    cleanup_step as clean_step
)

@pipeline
def scrape_pipeline(cfg_path: str):
    cfg = load_config_step(cfg_path)
    scrape_cfg = scrape_step(cfg)
    clean_step(scrape_cfg)
    cleanup_step(scrape_cfg)

if __name__ == "__main__":
    scrape_pipeline(cfg_path="/home/mindmap/Desktop/SLM/unstructured_data/config.yml")