from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import (
    jina_health_check,
    groq_health_check,
    load_config,
    display_config
)
from config import (
    HealthResponse,
    HealthRequest,

    ScrapeRequest,
    ScrapeResponse,
    ScrapeConfig,
)
from scrape_flow import ScrapeUrl
from pathlib import Path
import os

## config
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(BASE_DIR, "config.yml")
except Exception as e:
    logging.error(e)

app = FastAPI()

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://0.0.0.0:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/health")
def health_check(requests: HealthRequest):
    return HealthResponse(
        groq_isActive=groq_health_check(api_key=requests.groq_api_key),
        jina_isActive=jina_health_check(api_key=requests.jina_api_key)
    )

@app.post("/scraper/jobs")
def scrape_website(request: ScrapeRequest):
    try:
        cfg = load_config(CONFIG_PATH)
        cfg = ScrapeConfig(**cfg['web_scrapping'])
        cfg.url = request.url
        cfg.api_key = request.api_key
        cfg.scrape_type = request.scrape_type
        cfg.max_depth = request.max_depth
        cfg.max_workers = request.max_workers
        display_config(cfg)
        scrape_obj = ScrapeUrl(cfg)
        scrape_obj.scrape_url()
        scrape_obj.clean_markdown()
        return ScrapeResponse(
            job_id = None,
            status = "completed",
            
        )
    except Exception as e:
        return ScrapeResponse(
            job_id = None,
            status = "failed",
            number_of_links_scrapped = None,
            created_at = None,
            message = str(e)
        )
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)